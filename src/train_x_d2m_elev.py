import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_x_d2m_elev import DownscaleDatasetXD2MElev
from model_cnn import UNetDownscale


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
    return (diff ** 2).mean()


def run_epoch(model, loader, optimizer, device, train=True):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_count = 0

    for x, y, mask in loader:
        x = x.to(device)
        y = y.to(device)
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
    base_train_path = ".data/downscaling_splits/train_norm.nc"
    base_val_path = ".data/downscaling_splits/val_norm.nc"

    extra_train_path = ".data/downscaling_splits_extra/train_extra_norm.nc"
    extra_val_path = ".data/downscaling_splits_extra/val_extra_norm.nc"

    topo_path = ".data/ETOPO2/topography_features_on_gridmet_masked_norm.nc"

    batch_size = 16
    lr = 1e-4
    n_epochs = 20
    num_workers = 0

    os.makedirs("outputs/checkpoints", exist_ok=True)

    save_path = "outputs/checkpoints/best_unet_x_d2m_elev.pt"
    history_path = "outputs/x_d2m_elev_loss_history.csv"

    device = get_device()
    print("Using device:", device)

    train_ds = DownscaleDatasetXD2MElev(base_train_path, extra_train_path, topo_path)
    val_ds = DownscaleDatasetXD2MElev(base_val_path, extra_val_path, topo_path)

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = UNetDownscale(in_channels=3).to(device)
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

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    history = pd.DataFrame({
        "epoch": list(range(1, n_epochs + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
    })
    history.to_csv(history_path, index=False)
    print(f"Saved loss history to {history_path}")

    print("Training finished.")
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()