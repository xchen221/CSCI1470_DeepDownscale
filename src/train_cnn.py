import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import DownscaleDataset
from model_cnn import UNetDownscale  # Changed from SimpleCNN


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
    if train:
        model.train()
    else:
        model.eval()

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
    train_path = ".data/downscaling_splits/train_norm.nc"
    val_path = ".data/downscaling_splits/val_norm.nc"
    topo_path = ".data/ETOPO2/topography_features_on_gridmet_masked_norm.nc"

    batch_size = 32
    lr = 3e-4
    n_epochs = 20

    os.makedirs("outputs/checkpoints", exist_ok=True)
    save_path = "outputs/checkpoints/best_cnn.pt"
    history_path = "outputs/cnn_loss_history.csv"

    device = get_device()
    print("Using device:", device)

    train_ds = DownscaleDataset(train_path, topo_path)
    val_ds = DownscaleDataset(val_path, topo_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    #model = SimpleCNN(in_channels=2).to(device)
    # Old: model = SimpleCNN(in_channels=2).to(device)
    model = UNetDownscale(in_channels=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.2)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_loss) # Step the scheduler based on validation loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

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