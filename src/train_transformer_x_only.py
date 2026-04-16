import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_x_only import DownscaleDatasetXOnly
from model_transformer import UNetTransformerDownscale


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def masked_mse_loss(pred, target, mask):
    valid = (mask > 0.5) & torch.isfinite(pred) & torch.isfinite(target)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    diff = pred[valid] - target[valid]
    return (diff**2).mean()


def run_epoch(model, loader, optimizer, device, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_count = 0

    for x, y, mask in loader:
        x = x.float().to(device).contiguous()
        y = y.float().to(device)
        mask = mask.to(device)

        batch_size = x.size(0)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            pred = model(x)
            loss = masked_mse_loss(pred, y, mask)

            if train:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count


def main():
    train_path = ".data/downscaling_splits/train_norm.nc"
    val_path = ".data/downscaling_splits/val_norm.nc"

    batch_size = 4
    lr = 1e-4
    n_epochs = 30

    os.makedirs("outputs/checkpoints", exist_ok=True)
    save_path = "outputs/checkpoints/best_transformer_x_only_oscar.pt"
    history_path = "outputs/transformer_x_only_oscar_loss_history.csv"

    device = get_device()
    print("Using device:", device)

    train_ds = DownscaleDatasetXOnly(train_path)
    val_ds = DownscaleDatasetXOnly(val_path)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    model = (
        UNetTransformerDownscale(
            in_channels=1,   # T only
            out_channels=1,
            base_channels=32,
            bottleneck_channels=128,
            embed_dim=256,
            num_heads=8,
            num_layers=3,
            dropout=0.1,
        )
        .float()
        .to(device)
    )

    # sanity check
    x0, y0, m0 = next(iter(train_loader))
    x0 = x0.float().to(device).contiguous()

    print("sanity x shape:", x0.shape)
    print("x0 dtype:", x0.dtype)
    print("x0 device:", x0.device)
    print("x0 contiguous:", x0.is_contiguous())
    print("x0 finite:", torch.isfinite(x0).all().item())
    print("x0 min/max:", x0.min().item(), x0.max().item())

    test_conv = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1).float().to(device)
    with torch.no_grad():
        z = test_conv(x0[:1])
    print("plain conv output shape:", z.shape)

    with torch.no_grad():
        yhat0 = model(x0[:1])
    print("sanity out shape:", yhat0.shape)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.2
    )

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    history = pd.DataFrame(
        {
            "epoch": list(range(1, n_epochs + 1)),
            "train_loss": train_losses,
            "val_loss": val_losses,
        }
    )
    history.to_csv(history_path, index=False)
    print(f"Saved loss history to {history_path}")

    print("Training finished.")
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()