from os import getenv
import torch
import chess
import chess.svg
import subprocess
import shutil
from training.common import config, BoardEncoder, ChessTransformer

# 加载模型
def load_model(path):
    model = ChessTransformer().to(config.device)
    model.load_state_dict(torch.load(path, map_location=config.device))
    model.eval()
    return model

# 生成AI走法
def generate_move(board, model, encoder):
    # 编码当前棋盘状态
    board_tensor = encoder.encode(board)
    input_tensor = torch.tensor(board_tensor, dtype=torch.float32)
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0).to(config.device)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(input_tensor)
        logits = outputs[0].cpu().numpy()
    
    # 筛选合法走法
    legal_moves = list(board.legal_moves)
    best_move = None
    best_score = -float('inf')
    
    for move in legal_moves:
        # 创建移动标签
        label = move.from_square * 64 + move.to_square
        score = logits[label]
        
        if score > best_score:
            best_score = score
            best_move = move
    
    # 处理兵升变
    if (best_move and board.piece_type_at(best_move.from_square) == chess.PAWN and
        chess.square_rank(best_move.to_square) in [0, 7]):
        return chess.Move(best_move.from_square, best_move.to_square, promotion=chess.QUEEN)
    
    return best_move

def print_board(board, arrow):
    with open("board.svg", 'w') as file:
        file.write(chess.svg.board(board, arrows=arrow and [arrow] or []))
    if getenv("TERM") == "xterm-kitty":
        subprocess.run("kitty +kitten icat board.svg".split())
    else:
        subprocess.run("wezterm imgcat board.svg".split())

# 主交互循环
def main():
    if getenv("TERM") == "xterm-kitty":
        assert(shutil.which("kitty"))
    else:
        assert(shutil.which("wezterm"))
    # 初始化组件
    encoder = BoardEncoder()
    model = load_model(config.model_save_path)
    board = chess.Board()
    
    print("国际象棋AI已启动 (输入'quit'退出)")
    print_board(board, None)
    
    while True:
        # 打印当前棋盘
        print("\n当前棋盘:")
        
        # 用户输入
        uci = input("\n你的走法 (UCI格式): ").strip()
        if uci.lower() == "quit":
            break
        
        # 执行用户走法
        try:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                print("非法走法! 请重试")
                continue
            
            board.push(move)
            
            # 检查游戏结束
            print_board(board, chess.svg.Arrow(move.from_square, move.to_square, color="green"))
            if board.is_game_over():
                print("游戏结束! 结果: ", board.result())
                break
            
            # AI生成走法
            ai_move = generate_move(board, model, encoder)
            if ai_move:
                board.push(ai_move)
                print(f"AI走法: {ai_move.uci()}")
                print_board(board, chess.svg.Arrow(ai_move.from_square, ai_move.to_square, color="green"))
                
                # 检查游戏结束
                if board.is_game_over():
                    print("游戏结束! 结果: ", board.result())
                    break
            else:
                print("AI无法生成合法走法!")
        except:
            print("无效输入! 请使用UCI格式 (如'e2e4')")

if __name__ == "__main__":
    # 导入必要的PyTorch模块
    
    main()
