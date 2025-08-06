import torch
import torch.nn.functional as F
import torch.nn as nn
import chess
import numpy as np
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 配置参数
class Config:
    # 数据参数
    data_path = "processed/"
    val_split = 0.1                # 验证集比例
    
    # 模型参数
    board_channels = 16             # 棋盘表示通道数
    d_model = 256                  # Transformer隐藏层维度
    nhead = 8                      # Transformer头数
    num_layers = 6                 # Transformer层数
    dim_feedforward = 512          # Transformer前馈网络维度
    
    # 训练参数
    batch_size = 64
    learning_rate = 1e-4
    num_epochs = 20
    weight_decay = 1e-5
    clip_value = 1.0               # 梯度裁剪值
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 输出路径
    model_save_path = "chess_transformer.pth"
    plot_save_path = "training_plot.png"

config = Config()

# 棋盘状态编码器
class BoardEncoder:
    def __init__(self):
        # 棋子类型映射
        self.piece_types = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5
        }
        # 特殊状态标志索引
        self.special_indices = {
            'white_kingside_castle': 6,
            'white_queenside_castle': 7,
            'black_kingside_castle': 8,
            'black_queenside_castle': 9,
            'en_passant': 10,
            'halfmove_clock': 11,
            'fullmove_number': 12,
            'side_to_move': 13,
            'in_check': 14,
            'checkmate': 15
        }
    
    def encode(self, board):
        """将棋盘状态编码为8x8x16的张量"""
        tensor = np.zeros((8, 8, config.board_channels), dtype=np.float32)
        
        # 编码棋子位置
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                row, col = 7 - square // 8, square % 8
                channel = self.piece_types[piece.piece_type]
                # 为白棋设置正值，黑棋设置负值
                tensor[row, col, channel] = 1.0 if piece.color == chess.WHITE else -1.0
        
        # 编码特殊状态
        tensor[:, :, self.special_indices['white_kingside_castle']] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        tensor[:, :, self.special_indices['white_queenside_castle']] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        tensor[:, :, self.special_indices['black_kingside_castle']] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        tensor[:, :, self.special_indices['black_queenside_castle']] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0
        
        # 吃过路兵位置
        if board.ep_square:
            ep_row, ep_col = 7 - board.ep_square // 8, board.ep_square % 8
            tensor[ep_row, ep_col, self.special_indices['en_passant']] = 1.0
        
        # 移动计数
        tensor[:, :, self.special_indices['halfmove_clock']] = board.halfmove_clock / 50.0
        tensor[:, :, self.special_indices['fullmove_number']] = board.fullmove_number / 100.0
        
        # 当前玩家
        tensor[:, :, self.special_indices['side_to_move']] = 1.0 if board.turn == chess.WHITE else -1.0
        
        # 将军状态
        tensor[:, :, self.special_indices['in_check']] = 1.0 if board.is_check() else 0.0
        tensor[:, :, self.special_indices['checkmate']] = 1.0 if board.is_checkmate() else 0.0
        
        return tensor
    
    def decode_move(self, label, board):
        """将标签解码为国际象棋移动"""
        # 起始位置和目标位置
        from_square = label // 64
        to_square = label % 64
        
        # 创建移动对象
        move = chess.Move(from_square, to_square)
        
        # 处理升变
        if move in board.legal_moves and board.piece_type_at(from_square) == chess.PAWN:
            if chess.square_rank(to_square) in [0, 7]:  # 兵到达对方底线
                move = chess.Move(from_square, to_square, promotion=chess.QUEEN)  # 默认升变为皇后
        
        return move

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual  # 残差连接
        return F.relu(out)

# Transformer模型
class ChessTransformer(nn.Module):
    def __init__(self):
        super(ChessTransformer, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(config.board_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # 插入残差块（替换原第二卷积层）
            ResidualBlock(64),

            nn.MaxPool2d(2, 2),  # 下采样到4x4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # 注意通道数变化
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # 在高层特征再次插入残差
            ResidualBlock(128),

            nn.Conv2d(128, config.d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(config.d_model),
            nn.ReLU()
        )
        
        # 位置编码
        self.pos_embedding = nn.Parameter(torch.randn(16, config.d_model))  # 4x4=16个位置
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, config.num_layers)
        
        # 输出层
        self.output_head = nn.Sequential(
            nn.Linear(config.d_model, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 4096)  # 64x64=4096种可能移动
        )
    
    def forward(self, x):
        # 卷积特征提取
        x = self.conv_block(x)  # (batch, d_model, 4, 4)
        
        # 重塑为序列 (batch, seq_len, d_model)
        batch_size = x.size(0)
        x = x.view(batch_size, config.d_model, -1).permute(0, 2, 1)  # (batch, 16, d_model)
        
        # 添加位置编码
        x = x + self.pos_embedding.unsqueeze(0)
        
        # Transformer处理
        x = self.transformer(x)
        
        # 平均池化
        x = x.mean(dim=1)
        
        # 输出层
        return self.output_head(x)
