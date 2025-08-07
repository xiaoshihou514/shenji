import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.types import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import chess
from tqdm import tqdm
import matplotlib.pyplot as plt
from training.common import config, BoardEncoder, ChessTransformer

# 数据集类
class ChessDataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(self, games, encoder):
        self.encoder = encoder
        self.positions = []
        self.labels = []
        
        print("Processing games...")
        for game in tqdm(games):
            board = chess.Board()
            moves = game["moves"]
            
            for move_uci in moves:
                # 创建移动对象
                move = chess.Move.from_uci(move_uci)
                
                # 跳过非法移动
                assert(move in board.legal_moves)
                
                # 编码棋盘状态
                board_tensor = self.encoder.encode(board)
                
                # 编码移动为标签 (64*64=4096种可能移动)
                label = move.from_square * 64 + move.to_square
                
                self.positions.append(board_tensor)
                self.labels.append(label)
                
                # 执行移动
                board.push(move)
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        position = torch.tensor(self.positions[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return position.permute(2, 0, 1), label  # 通道优先格式


# 训练函数
def train(model, train_loader, val_loader, optimizer, criterion, scheduler):
    train_losses = []
    val_losses = []
    accuracies = []
    
    best_val_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_value)
            
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(config.device), labels.to(config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                
                # 计算准确率
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        accuracy = correct / total
        accuracies.append(accuracy)
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.model_save_path)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        
        print(f"Epoch {epoch+1}/{config.num_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Accuracy: {accuracy:.4f}")
    
    return train_losses, val_losses, accuracies

# 绘制训练曲线
def plot_training_curves(train_losses, val_losses, accuracies):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(config.plot_save_path)
    plt.show()

# 主函数
def main():
    print("start")
    # 设置随机种子以确保可复现性
    torch.manual_seed(42)
    np.random.seed(42)

    print(f"Is CUDA supported by this system? {torch.cuda.is_available()}")

    # Storing ID of current CUDA device
    cuda_id = torch.cuda.current_device()
    print(f"ID of current CUDA device: {torch.cuda.current_device()}")
          
    print(f"Name of current CUDA device: {torch.cuda.get_device_name(cuda_id)}")
    # 加载游戏数据
    if not os.path.exists(config.data_path):
        print(f"Error: Data file not found at {config.data_path}")
        print("Please provide chess game data in JSON format.")
        return
    
    games = []
    
    # 遍历文件夹中的所有文件
    print("start loading")
    i: Int = 0
    for filename in os.listdir(config.data_path):
        file_path = os.path.join(config.data_path, filename)
        
        try:
            # 读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                games.append(data)
                i += 1
                    
        except json.JSONDecodeError as e:
            print(f"错误: 文件 {filename} 不是有效的JSON格式: {e}")
        except Exception as e:
            print(f"错误: 读取文件 {filename} 时出错: {e}")

        if i % 1000 == 0:
            print(f"Loaded {i} games")
    
    print(f"Loaded {len(games)} games")
    
    # 初始化编码器
    encoder = BoardEncoder()
    
    # 创建数据集
    dataset = ChessDataset(games, encoder)
    
    # 分割数据集
    val_size = int(len(dataset) * config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # 初始化模型
    print(config.device)
    model = ChessTransformer().to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # 训练模型
    train_losses, val_losses, accuracies = train(model, train_loader, val_loader, 
                                               optimizer, criterion, scheduler)
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, accuracies)
    
    print("Training completed!")

if __name__ == "__main__":
    main()
