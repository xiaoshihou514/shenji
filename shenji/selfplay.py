"""
selfplay.py – generate RL replay data via batched self-play or model-vs-opponent games.

Usage:
    uv run shenji/selfplay.py --checkpoint checkpoints/best.pt --out ./rl_replay.npz

The replay format is a `.npz` file with:
    x       : uint8   (N, 71)   encoded board states
    y       : int16   (N,)      chosen move indices
    returns : float32 (N,)      discounted returns from the mover's perspective
    fen     : <U96    (N,)      FEN before the move (used to rebuild legal masks)
    color   : int8    (N,)      1 for white mover, 0 for black mover
    source  : <U16    (N,)      "self", "history", or "engine"
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import chess.engine
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from shenji.board import BoardEncoder, MoveCodec
from shenji.model import ChessTransformer

__all__ = [
    "ReplayDataset",
    "concat_replays",
    "evaluate_matches",
    "generate_replay",
    "load_policy_model",
    "save_replay",
]


@dataclass(slots=True)
class _RecordedStep:
    x: np.ndarray
    fen: str
    move_idx: int
    color: bool
    ply: int
    source: str


@dataclass(slots=True)
class _GameSlot:
    board: chess.Board
    current_color: bool | None
    steps: list[_RecordedStep]
    source: str
    repetition_offender: bool | None = None


class ReplayDataset(Dataset):
    """Torch dataset for RL replay files."""

    def __init__(self, replay: dict[str, np.ndarray]) -> None:
        self.x = replay["x"]
        self.y = replay["y"]
        self.returns = replay["returns"]
        self.fen = replay["fen"]

    @classmethod
    def from_npz(cls, path: Path) -> "ReplayDataset":
        npz = np.load(path, allow_pickle=False)
        replay = {
            "x": npz["x"],
            "y": npz["y"],
            "returns": npz["returns"],
            "fen": npz["fen"],
        }
        return cls(replay)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.x[idx].astype(np.int64, copy=False))
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        ret = torch.tensor(float(self.returns[idx]), dtype=torch.float32)
        return x, y, ret, str(self.fen[idx])


def _amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_policy_model(checkpoint: Path, device: torch.device) -> ChessTransformer:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    saved_cfg = state.get("cfg", {})
    model = ChessTransformer(
        d_model=saved_cfg.get("d_model", 768),
        nhead=saved_cfg.get("nhead", 12),
        num_layers=saved_cfg.get("num_layers", 12),
        dim_feedforward=saved_cfg.get("dim_feedforward", 3072),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    return model


def save_replay(path: Path, replay: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **replay)


def concat_replays(replays: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not replays:
        raise ValueError("replays list must not be empty")
    keys = replays[0].keys()
    return {key: np.concatenate([r[key] for r in replays], axis=0) for key in keys}


def _temperature_for_ply(
    ply: int,
    temperature_open: float,
    temperature_mid: float,
    temperature_switch_ply: int,
) -> float:
    return temperature_open if ply < temperature_switch_ply else temperature_mid


def _pick_moves_batch(
    model: ChessTransformer,
    boards: list[chess.Board],
    device: torch.device,
    deterministic: bool,
    temperature_open: float,
    temperature_mid: float,
    temperature_switch_ply: int,
    dirichlet_alpha: float | None,
    dirichlet_eps: float,
) -> list[tuple[int, chess.Move]]:
    if not boards:
        return []

    x = torch.stack([BoardEncoder.encode(board) for board in boards]).to(device)
    amp_dtype = _amp_dtype(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
            logits = model(x)
    logits = logits.float()

    picks: list[tuple[int, chess.Move]] = []
    for row, board in zip(logits, boards, strict=True):
        mask = MoveCodec.legal_mask(board).to(row.device)
        masked = row.masked_fill(~mask, -torch.inf)
        if deterministic or mask.sum().item() <= 1:
            idx = int(masked.argmax())
        else:
            temperature = _temperature_for_ply(
                board.ply(),
                temperature_open=temperature_open,
                temperature_mid=temperature_mid,
                temperature_switch_ply=temperature_switch_ply,
            )
            scaled = masked if temperature <= 0 else masked / temperature
            probs = torch.softmax(scaled, dim=0)
            legal_idx = mask.nonzero(as_tuple=False).squeeze(1)
            if dirichlet_alpha and dirichlet_eps > 0.0 and board.ply() < temperature_switch_ply:
                legal_probs = probs[legal_idx].cpu().numpy()
                noise = np.random.dirichlet(
                    np.full(len(legal_idx), dirichlet_alpha, dtype=np.float64)
                )
                mixed = (1.0 - dirichlet_eps) * legal_probs + dirichlet_eps * noise
                mixed = mixed / mixed.sum()
                idx = int(np.random.choice(legal_idx.cpu().numpy(), p=mixed))
            else:
                idx = int(torch.multinomial(probs, 1))
        picks.append((idx, MoveCodec.to_move(idx, board)))
    return picks


def _new_slot(game_idx: int, source: str) -> _GameSlot:
    current_color = None if source == "self" else (chess.WHITE if game_idx % 2 == 0 else chess.BLACK)
    return _GameSlot(board=chess.Board(), current_color=current_color, steps=[], source=source)


def _white_result(
    board: chess.Board,
    max_moves_hit: bool,
    repetition_offender: bool | None,
    repetition_penalty: float,
) -> float:
    if max_moves_hit:
        return 0.0
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        result = 0.0
    else:
        result = 1.0 if outcome.winner == chess.WHITE else -1.0
    if result == 0.0 and repetition_penalty > 0.0 and repetition_offender is not None:
        return -repetition_penalty if repetition_offender == chess.WHITE else repetition_penalty
    return result


def _initial_replay() -> dict[str, list[Any]]:
    return {"x": [], "y": [], "returns": [], "fen": [], "color": [], "source": []}


def _finalise_replay(replay: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    if not replay["x"]:
        empty_u96 = np.empty((0,), dtype="<U96")
        empty_u16 = np.empty((0,), dtype="<U16")
        return {
            "x": np.empty((0, 71), dtype=np.uint8),
            "y": np.empty((0,), dtype=np.int16),
            "returns": np.empty((0,), dtype=np.float32),
            "fen": empty_u96,
            "color": np.empty((0,), dtype=np.int8),
            "source": empty_u16,
        }

    max_fen = max(len(fen) for fen in replay["fen"])
    max_source = max(len(src) for src in replay["source"])
    return {
        "x": np.stack(replay["x"]).astype(np.uint8, copy=False),
        "y": np.array(replay["y"], dtype=np.int16),
        "returns": np.array(replay["returns"], dtype=np.float32),
        "fen": np.array(replay["fen"], dtype=f"<U{max(96, max_fen)}"),
        "color": np.array(replay["color"], dtype=np.int8),
        "source": np.array(replay["source"], dtype=f"<U{max(16, max_source)}"),
    }


def _run_match_loop(
    model: ChessTransformer,
    device: torch.device,
    games: int,
    batch_size: int,
    max_moves: int,
    temperature_open: float,
    temperature_mid: float,
    temperature_switch_ply: int,
    gamma: float,
    repetition_penalty: float,
    *,
    collect_replay: bool,
    source: str,
    deterministic_current: bool,
    opponent_model: ChessTransformer | None = None,
    deterministic_opponent: bool = True,
    engine_path: Path | None = None,
    engine_depth: int = 5,
    dirichlet_alpha: float | None = None,
    dirichlet_eps: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if opponent_model is not None and engine_path is not None:
        raise ValueError("opponent_model and engine_path are mutually exclusive")

    replay = _initial_replay()
    summary = {"wins": 0.0, "draws": 0.0, "losses": 0.0, "avg_plies": 0.0}
    slots = [_new_slot(i, source) for i in range(min(batch_size, games))]
    launched = len(slots)
    total_plies = 0
    engine = None
    if engine_path is not None:
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))

    pbar = tqdm(total=games, desc=f"{source} games", unit="game")
    try:
        while slots:
            current_slots_idx: list[int] = []
            current_boards: list[chess.Board] = []
            opponent_slots_idx: list[int] = []
            opponent_boards: list[chess.Board] = []

            for idx, slot in enumerate(slots):
                if slot.current_color is None or slot.board.turn == slot.current_color:
                    current_slots_idx.append(idx)
                    current_boards.append(slot.board)
                elif opponent_model is not None:
                    opponent_slots_idx.append(idx)
                    opponent_boards.append(slot.board)

            current_moves = _pick_moves_batch(
                model,
                current_boards,
                device,
                deterministic=deterministic_current,
                temperature_open=temperature_open,
                temperature_mid=temperature_mid,
                temperature_switch_ply=temperature_switch_ply,
                dirichlet_alpha=dirichlet_alpha if collect_replay else None,
                dirichlet_eps=dirichlet_eps if collect_replay else 0.0,
            )
            for slot_idx, (move_idx, move) in zip(current_slots_idx, current_moves, strict=True):
                slot = slots[slot_idx]
                if collect_replay:
                    slot.steps.append(
                        _RecordedStep(
                            x=BoardEncoder.encode_np(slot.board),
                            fen=slot.board.fen(),
                            move_idx=move_idx,
                            color=slot.board.turn,
                            ply=slot.board.ply(),
                            source=slot.source,
                        )
                    )
                mover = slot.board.turn
                slot.board.push(move)
                if slot.board.can_claim_threefold_repetition() or slot.board.can_claim_fifty_moves():
                    slot.repetition_offender = mover

            opponent_moves = _pick_moves_batch(
                opponent_model,
                opponent_boards,
                device,
                deterministic=deterministic_opponent,
                temperature_open=0.0,
                temperature_mid=0.0,
                temperature_switch_ply=0,
                dirichlet_alpha=None,
                dirichlet_eps=0.0,
            ) if opponent_model is not None else []
            for slot_idx, (_, move) in zip(opponent_slots_idx, opponent_moves, strict=True):
                slot = slots[slot_idx]
                mover = slot.board.turn
                slot.board.push(move)
                if slot.board.can_claim_threefold_repetition() or slot.board.can_claim_fifty_moves():
                    slot.repetition_offender = mover
            if engine is not None:
                for idx, slot in enumerate(slots):
                    # 跳过模型自己走棋的 slot
                    if slot.current_color is None or slot.board.turn == slot.current_color:
                        continue

                    # 如果游戏已经结束，就不再让引擎走棋，让它留在下一轮被终结
                    if slot.board.is_game_over(claim_draw=True):
                        continue

                    mover = slot.board.turn
                    try:
                        result = engine.play(slot.board, chess.engine.Limit(depth=engine_depth))
                        move = result.move
                    except Exception:
                        move = None

                    if move is None:
                        # 引擎在合法局面下也可能会返回 None（极少见），此时直接判该局为和棋
                        # 简便做法：直接视为“强制结束”，把该 slot 的 board 推进到满步数
                        # 下面的 max_moves_hit 条件会在后续循环捕获它
                        print(f"⚠ Engine returned None, forcing draw by max moves.")
                        # 塞一个合法走法来保证 ply 增加（选第一个合法走法即可）
                        legal = list(slot.board.generate_legal_moves())
                        if legal:
                            slot.board.push(legal[0])
                        # 如果连合法走法都没有，说明已经结束，什么都不做
                    else:
                        slot.board.push(move)

                    if slot.board.can_claim_threefold_repetition() or slot.board.can_claim_fifty_moves():
                        slot.repetition_offender = mover

            next_slots: list[_GameSlot] = []
            for slot in slots:
                max_moves_hit = slot.board.ply() >= max_moves
                draw_claim = slot.board.can_claim_threefold_repetition() or slot.board.can_claim_fifty_moves()
                if not max_moves_hit and not draw_claim and not slot.board.is_game_over(claim_draw=True):
                    next_slots.append(slot)
                    continue

                white_result = _white_result(
                    slot.board,
                    max_moves_hit=max_moves_hit,
                    repetition_offender=slot.repetition_offender,
                    repetition_penalty=repetition_penalty,
                )
                total_plies += slot.board.ply()
                if white_result > 0:
                    summary["wins"] += 1 if slot.current_color in (None, chess.WHITE) else 0
                    summary["losses"] += 1 if slot.current_color == chess.BLACK else 0
                elif white_result < 0:
                    summary["wins"] += 1 if slot.current_color == chess.BLACK else 0
                    summary["losses"] += 1 if slot.current_color in (None, chess.WHITE) else 0
                else:
                    summary["draws"] += 1

                final_ply = slot.board.ply()
                for step in slot.steps:
                    outcome = white_result if step.color == chess.WHITE else -white_result
                    ret = (gamma ** max(final_ply - 1 - step.ply, 0)) * outcome
                    replay["x"].append(step.x)
                    replay["y"].append(step.move_idx)
                    replay["returns"].append(ret)
                    replay["fen"].append(step.fen)
                    replay["color"].append(1 if step.color == chess.WHITE else 0)
                    replay["source"].append(step.source)

                pbar.update(1)
                if launched < games:
                    next_slots.append(_new_slot(launched, source))
                    launched += 1
            slots = next_slots
    finally:
        if engine is not None:
            engine.quit()
        pbar.close()

    summary["avg_plies"] = total_plies / max(games, 1)
    return _finalise_replay(replay), summary


def generate_replay(
    model: ChessTransformer,
    device: torch.device,
    games: int,
    batch_size: int,
    max_moves: int,
    gamma: float,
    temperature_open: float,
    temperature_mid: float,
    temperature_switch_ply: int,
    *,
    source: str = "self",
    opponent_model: ChessTransformer | None = None,
    engine_path: Path | None = None,
    engine_depth: int = 5,
    dirichlet_alpha: float | None = None,
    dirichlet_eps: float = 0.0,
    repetition_penalty: float = 0.0,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    return _run_match_loop(
        model,
        device,
        games=games,
        batch_size=batch_size,
        max_moves=max_moves,
        temperature_open=temperature_open,
        temperature_mid=temperature_mid,
        temperature_switch_ply=temperature_switch_ply,
        gamma=gamma,
        repetition_penalty=repetition_penalty,
        collect_replay=True,
        source=source,
        deterministic_current=False,
        opponent_model=opponent_model,
        deterministic_opponent=True,
        engine_path=engine_path,
        engine_depth=engine_depth,
        dirichlet_alpha=dirichlet_alpha,
        dirichlet_eps=dirichlet_eps,
    )


def evaluate_matches(
    model: ChessTransformer,
    device: torch.device,
    games: int,
    batch_size: int,
    max_moves: int,
    *,
    opponent_model: ChessTransformer | None = None,
    engine_path: Path | None = None,
    engine_depth: int = 5,
) -> dict[str, float]:
    _, summary = _run_match_loop(
        model,
        device,
        games=games,
        batch_size=batch_size,
        max_moves=max_moves,
        temperature_open=0.0,
        temperature_mid=0.0,
        temperature_switch_ply=0,
        gamma=1.0,
        repetition_penalty=0.0,
        collect_replay=False,
        source="eval",
        deterministic_current=True,
        opponent_model=opponent_model,
        deterministic_opponent=True,
        engine_path=engine_path,
        engine_depth=engine_depth,
        dirichlet_alpha=None,
        dirichlet_eps=0.0,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate self-play replay data")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-moves", type=int, default=300)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--temperature-open", type=float, default=1.0)
    parser.add_argument("--temperature-mid", type=float, default=0.3)
    parser.add_argument("--temperature-switch-ply", type=int, default=10)
    parser.add_argument("--dirichlet-alpha", type=float, default=None)
    parser.add_argument("--dirichlet-eps", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=0.0)
    parser.add_argument("--opponent-checkpoint", type=Path, default=None)
    parser.add_argument("--engine-path", type=Path, default=None)
    parser.add_argument("--engine-depth", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.opponent_checkpoint and args.engine_path:
        raise SystemExit("Use either --opponent-checkpoint or --engine-path, not both.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_policy_model(args.checkpoint, device)
    opponent_model = (
        load_policy_model(args.opponent_checkpoint, device)
        if args.opponent_checkpoint
        else None
    )
    source = "engine" if args.engine_path else ("history" if args.opponent_checkpoint else "self")
    replay, summary = generate_replay(
        model,
        device,
        games=args.games,
        batch_size=args.batch_size,
        max_moves=args.max_moves,
        gamma=args.gamma,
        temperature_open=args.temperature_open,
        temperature_mid=args.temperature_mid,
        temperature_switch_ply=args.temperature_switch_ply,
        source=source,
        opponent_model=opponent_model,
        engine_path=args.engine_path,
        engine_depth=args.engine_depth,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        repetition_penalty=args.repetition_penalty,
    )
    save_replay(args.out, replay)
    stats = {
        "replay_path": str(args.out),
        "games": args.games,
        "samples": int(replay["y"].shape[0]),
        "wins": int(summary["wins"]),
        "draws": int(summary["draws"]),
        "losses": int(summary["losses"]),
        "avg_plies": round(summary["avg_plies"], 2),
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
