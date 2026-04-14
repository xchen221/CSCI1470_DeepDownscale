import os

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from dataset_window import DownscaleWindowDataset
from model_unet import UNet


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


def run_epoch(model, loader, optimizer, device, train=True, epoch=None, n_epochs=None):
    if train:
        model.train()
        desc = f"Train {epoch:02d}/{n_epochs:02d}"
    else:
        model.eval()
        desc = f"Val   {epoch:02d}/{n_epochs:02d}"

    total_loss = 0.0
    total_count = 0

    pbar = tqdm(loader, desc=desc, leave=False)

    for x, y, mask in pbar:
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

        avg_loss = total_loss / total_count
        pbar.set_postfix(batch_loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}")

    return total_loss / total_count


def main():
    train_path = ".data/downscaling_splits/train_norm.nc"
    val_path = ".data/downscaling_splits/val_norm.nc"

    window_size = 3
    use_elev = True
    use_date = True

    in_channels = window_size + int(use_elev) + 2 * int(use_date)

    batch_size = 4
    lr = 1e-3
    n_epochs = 20

    os.makedirs("outputs/checkpoints", exist_ok=True)
    save_path = "outputs/checkpoints/best_unet_w3_elev_date.pt"

    device = get_device()
    print("Using device:", device)

    train_ds = DownscaleWindowDataset(
        train_path,
        window_size=window_size,
        use_elev=use_elev,
        use_date=use_date,
    )
    val_ds = DownscaleWindowDataset(
        val_path,
        window_size=window_size,
        use_elev=use_elev,
        use_date=use_date,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = UNet(in_channels=in_channels, out_channels=1, base_ch=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_loss = run_epoch(model, train_loader, optimizer, device, train=True, epoch=epoch, n_epochs=n_epochs)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False, epoch=epoch, n_epochs=n_epochs)

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")

    print("Training finished.")
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()