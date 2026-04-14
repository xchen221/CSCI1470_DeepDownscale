import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import DownscaleDataset
from model_cnn import SimpleCNN


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def masked_mae(pred, target, mask):
    valid = mask > 0.5
    return np.abs(pred[valid] - target[valid]).mean()


def masked_rmse(pred, target, mask):
    valid = mask > 0.5
    return np.sqrt(((pred[valid] - target[valid]) ** 2).mean())


def main():
    train_path = ".data/downscaling_splits/train_norm.nc"
    val_path = ".data/downscaling_splits/val_norm.nc"
    topo_path = ".data/ETOPO2/topography_features_on_gridmet_masked_norm.nc"

    batch_size = 4
    lr = 1e-3
    n_epochs = 20

    os.makedirs("outputs/checkpoints", exist_ok=True)
    save_path = "outputs/checkpoints/best_cnn.pt"

    device = get_device()
    print("Using device:", device)

    train_ds = DownscaleDataset(train_path, topo_path)
    val_ds = DownscaleDataset(val_path, topo_path)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = SimpleCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print("Training finished.")
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()