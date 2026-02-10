from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.data import MarketDataset
from src.logger import UnitaryLogger
from src.model import ModelConfig, UnitaryTransformer
from src.self_state import PositionInfo, PsychoConfig, PsychoModule, SelfState


@dataclass
class TrainConfig:
    data_path: str
    seq_len: int
    batch_size: int
    num_workers: int
    epochs: int
    lr: float
    weight_decay: float
    weight_price: float
    weight_self: float
    self_dim: int
    d_model: int
    n_heads: int
    n_layers: int
    dropout: float
    max_seq_len: int
    use_flash: bool
    compile_model: bool
    compile_mode: str
    log_interval: int
    max_steps: int | None
    log_dir: str
    log_color: bool
    initial_balance: float
    leverage: float
    cortisol_decay: float
    flat_cortisol_decay: float
    pain_scale: float
    stress_scale: float


ACTION_SIZES: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, -1.0)


def self_state_from_tensor(vec: torch.Tensor) -> SelfState:
    if vec.ndim == 1:
        vec = vec.unsqueeze(0)
    return SelfState(
        balance=vec[:, 0],
        exposure=vec[:, 1],
        pnl_unrealized=vec[:, 2],
        pain_distance=vec[:, 3],
        dopamine=vec[:, 4],
        cortisol=vec[:, 5],
    )


def build_dataloader(
    data_path: str,
    seq_len: int,
    batch_size: int,
    num_workers: int,
) -> tuple[MarketDataset, DataLoader]:
    dataset = MarketDataset(
        data_path,
        seq_len=seq_len + 1,
        normalize="log_return",
        return_raw=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return dataset, loader


def compute_action_targets(
    window_raw: torch.Tensor,
    entry_price: torch.Tensor,
    prev_state: SelfState,
    action_sizes: torch.Tensor,
    flat_steps: torch.Tensor,
    *,
    config: TrainConfig,
    psycho: PsychoModule,
) -> torch.Tensor:
    batch = window_raw.shape[0]
    action_sizes = action_sizes.to(device=window_raw.device, dtype=window_raw.dtype)
    action_sizes = action_sizes.view(1, -1).expand(batch, -1)
    flat_steps = flat_steps.to(device=window_raw.device, dtype=window_raw.dtype)
    flat_steps = flat_steps.view(batch, 1).expand_as(action_sizes)
    next_flat = torch.where(
        action_sizes == 0.0, flat_steps + 1.0, torch.zeros_like(flat_steps)
    )
    current_balance = config.initial_balance * prev_state.balance
    position_size = current_balance.unsqueeze(1) * action_sizes
    entry = entry_price.unsqueeze(1).expand_as(action_sizes)
    flat_window = window_raw.repeat_interleave(action_sizes.shape[1], dim=0)
    flat_balance = current_balance.repeat_interleave(action_sizes.shape[1])
    flat_steps_flat = next_flat.reshape(-1)
    position = PositionInfo(
        initial_balance=config.initial_balance,
        balance=flat_balance,
        position_size=position_size.reshape(-1),
        entry_price=entry.reshape(-1),
        leverage=config.leverage,
    )
    expected_pnl = prev_state.pnl_unrealized.repeat_interleave(action_sizes.shape[1])
    prev_cortisol = prev_state.cortisol.repeat_interleave(action_sizes.shape[1])
    target_state = psycho(
        flat_window,
        position,
        expected_pnl=expected_pnl,
        prev_cortisol=prev_cortisol,
        flat_steps=flat_steps_flat,
    )
    return target_state.as_tensor().view(batch, action_sizes.shape[1], -1)


def train(config: TrainConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        torch.set_float32_matmul_precision("high")

    dataset, loader = build_dataloader(
        config.data_path,
        config.seq_len,
        config.batch_size,
        config.num_workers,
    )
    feature_map = {name: idx for idx, name in enumerate(dataset.features)}
    required = {"close"}
    if not required.issubset(feature_map):
        raise ValueError(f"Dataset features missing required columns: {required}")
    model_config = ModelConfig(
        market_dim=len(dataset.features),
        self_dim=config.self_dim,
        action_dim=len(ACTION_SIZES),
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        dropout=config.dropout,
        max_seq_len=config.max_seq_len,
        use_flash=config.use_flash,
    )
    model = UnitaryTransformer(model_config).to(device)
    if config.compile_model:
        model = torch.compile(model, mode=config.compile_mode)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(0.9, 0.95),
        weight_decay=config.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    close_idx = feature_map["close"]
    psycho = PsychoModule(
        close_idx,
        PsychoConfig(
            cortisol_decay=config.cortisol_decay,
            flat_cortisol_decay=config.flat_cortisol_decay,
            pain_scale=config.pain_scale,
            stress_scale=config.stress_scale,
            inputs_are_log_returns=False,
        ),
    )
    action_sizes = torch.tensor(ACTION_SIZES, device=device)

    model.train()
    global_step = 0
    total_steps = len(loader) * config.epochs
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)
    print("total_steps", total_steps, "device", device)
    logger = UnitaryLogger(
        log_dir=config.log_dir,
        log_every=config.log_interval,
        enable_color=config.log_color,
    )
    try:
        carry_self: torch.Tensor | None = None
        flat_steps: torch.Tensor | None = None
        prev_position: torch.Tensor | None = None
        for epoch in range(config.epochs):
            for step, batch in enumerate(loader):
                if config.max_steps is not None and global_step >= config.max_steps:
                    return
                window, window_raw = batch
                window = window.to(device, non_blocking=True)
                window_raw = window_raw.to(device, non_blocking=True)
                inputs = window[:, :-1, :]
                targets = window[:, -1, :]
                inputs_raw = window_raw[:, :-1, :]
                entry_price = inputs_raw[:, -1, close_idx]
                if carry_self is None:
                    flat_position = PositionInfo(
                        initial_balance=config.initial_balance,
                        balance=config.initial_balance,
                        position_size=torch.zeros_like(entry_price),
                        entry_price=entry_price,
                        leverage=config.leverage,
                    )
                    prev_state = psycho(inputs_raw, flat_position)
                    prev_self = prev_state.as_tensor()
                    flat_steps = torch.zeros_like(entry_price)
                    prev_position = torch.zeros_like(entry_price)
                else:
                    prev_self = carry_self
                    prev_state = self_state_from_tensor(prev_self)
                    if flat_steps is None:
                        flat_steps = torch.zeros_like(entry_price)
                    if prev_position is None:
                        prev_position = torch.zeros_like(entry_price)
                target_self = compute_action_targets(
                    window_raw,
                    entry_price,
                    prev_state,
                    action_sizes,
                    flat_steps,
                    config=config,
                    psycho=psycho,
                )
                with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
                    action_logits, new_self, price_pred = model(
                        inputs, prev_self, return_price=True
                    )
                    action_probs = torch.softmax(action_logits, dim=-1)
                    balance_actual = prev_state.balance * config.initial_balance
                    expected_pos = (action_probs * action_sizes).sum(dim=-1) * balance_actual
                    curiosity_bonus = 0.001 * (expected_pos - prev_position).abs().mean()
                    price_loss = F.mse_loss(price_pred, targets)
                    self_loss = F.mse_loss(new_self, target_self)
                    loss = (
                        config.weight_price * price_loss
                        + config.weight_self * self_loss
                        - curiosity_bonus
                    )

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if global_step % config.log_interval == 0:
                    print(
                        "epoch",
                        epoch,
                        "step",
                        f"{global_step}/{total_steps}",
                        "loss",
                        f"{loss.item():.6f}",
                        "price",
                        f"{price_loss.item():.6f}",
                        "self",
                        f"{self_loss.item():.6f}",
                    )
                with torch.no_grad():
                    action_idx = torch.argmax(action_logits, dim=-1)
                    batch_indices = torch.arange(
                        action_idx.shape[0], device=action_idx.device
                    )
                    selected_state = target_self[batch_indices, action_idx, :]
                    carry_self = selected_state.detach()
                    action_vals = action_sizes[action_idx].float()
                    position_tensor = action_vals * prev_state.balance * config.initial_balance
                    position_size = float(position_tensor.mean().item())
                    price = float(entry_price.mean().item())
                    pnl = float(selected_state[:, 2].mean().item())
                    logger.log_step(
                        step=global_step,
                        price=price,
                        pnl=pnl,
                        self_state=selected_state.mean(dim=0),
                        position_size=position_size,
                    )
                    prev_position = position_tensor.detach()
                    flat_steps = torch.where(
                        action_vals == 0.0,
                        flat_steps + 1.0,
                        torch.zeros_like(flat_steps),
                    )
                global_step += 1
    finally:
        logger.close()


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train UnitaryTransformer.")
    parser.add_argument(
        "--data-path",
        default="data/BTCUSDT_2023-05-31_01-40_to_2025-04-24_12-19.json",
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--weight-price", type=float, default=1.0)
    parser.add_argument("--weight-self", type=float, default=1.0)
    parser.add_argument("--self-dim", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--use-flash", action="store_true", default=True)
    parser.add_argument("--no-flash", action="store_false", dest="use_flash")
    parser.add_argument("--compile", action="store_true", dest="compile_model", default=True)
    parser.add_argument("--no-compile", action="store_false", dest="compile_model")
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--log-color", action="store_true", default=True)
    parser.add_argument("--no-log-color", action="store_false", dest="log_color")
    parser.add_argument("--initial-balance", type=float, default=1.0)
    parser.add_argument("--leverage", type=float, default=3.0)
    parser.add_argument("--cortisol-decay", type=float, default=0.97)
    parser.add_argument("--flat-cortisol-decay", type=float, default=0.7)
    parser.add_argument("--pain-scale", type=float, default=80.0)
    parser.add_argument("--stress-scale", type=float, default=20.0)
    args = parser.parse_args()
    return TrainConfig(
        data_path=args.data_path,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        weight_price=args.weight_price,
        weight_self=args.weight_self,
        self_dim=args.self_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        use_flash=args.use_flash,
        compile_model=args.compile_model,
        compile_mode=args.compile_mode,
        log_interval=args.log_interval,
        max_steps=args.max_steps,
        log_dir=args.log_dir,
        log_color=args.log_color,
        initial_balance=args.initial_balance,
        leverage=args.leverage,
        cortisol_decay=args.cortisol_decay,
        flat_cortisol_decay=args.flat_cortisol_decay,
        pain_scale=args.pain_scale,
        stress_scale=args.stress_scale,
    )


def main() -> None:
    config = parse_args()
    if config.self_dim != 6:
        raise ValueError("self_dim must be 6 to match self-target definition.")
    if config.max_seq_len < config.seq_len:
        raise ValueError("max_seq_len must be >= seq_len.")
    train(config)


if __name__ == "__main__":
    main()
