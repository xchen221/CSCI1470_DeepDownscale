import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset_rich import DownscaleDatasetRich
from model_cnn import UNetDownscale


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def masked_rmse_torch(pred, target, mask):
    valid = (mask > 0.5) & torch.isfinite(pred) & torch.isfinite(target)
    if valid.sum() == 0:
        return float("nan")
    diff = pred[valid] - target[valid]
    return torch.sqrt((diff ** 2).mean()).item()


def evaluate_loader_rmse(model, loader, device):
    model.eval()
    all_pred = []
    all_target = []
    all_mask = []

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            pred = model(x)

            all_pred.append(pred.cpu())
            all_target.append(y.cpu())
            all_mask.append(mask.cpu())

    pred_all = torch.cat(all_pred, dim=0)
    target_all = torch.cat(all_target, dim=0)
    mask_all = torch.cat(all_mask, dim=0)

    return masked_rmse_torch(pred_all, target_all, mask_all)


def evaluate_permuted_rmse(model, loader, device, feature_idx, seed=42):
    model.eval()
    rng = np.random.default_rng(seed)

    all_pred = []
    all_target = []
    all_mask = []

    with torch.no_grad():
        for x, y, mask in loader:
            x = x.clone()  # do not modify original batch
            y = y.to(device)
            mask = mask.to(device)

            # permute this feature across batch samples
            perm = rng.permutation(x.size(0))
            x[:, feature_idx, :, :] = x[perm, feature_idx, :, :]

            x = x.to(device)
            pred = model(x)

            all_pred.append(pred.cpu())
            all_target.append(y.cpu())
            all_mask.append(mask.cpu())

    pred_all = torch.cat(all_pred, dim=0)
    target_all = torch.cat(all_target, dim=0)
    mask_all = torch.cat(all_mask, dim=0)

    return masked_rmse_torch(pred_all, target_all, mask_all)


def main():
    base_val_path = ".data/downscaling_splits/val_norm.nc"
    extra_val_path = ".data/downscaling_splits_extra/val_extra_norm.nc"
    topo_path = ".data/ETOPO2/topography_feature_stats_norm.nc"

    checkpoint_path = "outputs/checkpoints/best_unet_rich.pt"
    output_csv = "outputs/feature_importance_rich_val.csv"

    batch_size = 16
    num_workers = 0

    feature_names = [
        "x_norm",
        "ssrd_norm",
        "d2m_norm",
        "elevation_mean",
        "elevation_std",
        "slope_mean",
        "slope_std",
    ]

    device = get_device()
    print("Using device:", device)

    val_ds = DownscaleDatasetRich(
        base_nc_path=base_val_path,
        extra_nc_path=extra_val_path,
        topo_nc_path=topo_path,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = UNetDownscale(in_channels=7).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    print("Computing baseline validation RMSE...")
    baseline_rmse = evaluate_loader_rmse(model, val_loader, device)
    print(f"Baseline RMSE: {baseline_rmse:.6f}")

    results = []

    for i, feature_name in enumerate(feature_names):
        print(f"Permuting feature {i}: {feature_name}")
        permuted_rmse = evaluate_permuted_rmse(
            model, val_loader, device, feature_idx=i, seed=42
        )
        delta_rmse = permuted_rmse - baseline_rmse

        results.append(
            {
                "feature_idx": i,
                "feature_name": feature_name,
                "baseline_rmse": baseline_rmse,
                "permuted_rmse": permuted_rmse,
                "delta_rmse": delta_rmse,
            }
        )

        print(
            f"  permuted_rmse={permuted_rmse:.6f} | "
            f"delta_rmse={delta_rmse:.6f}"
        )

    df = pd.DataFrame(results).sort_values("delta_rmse", ascending=False)
    os.makedirs("outputs", exist_ok=True)
    df.to_csv(output_csv, index=False)

    print("\nFeature importance ranking (higher delta_rmse = more important):")
    print(df)
    print(f"\nSaved results to {output_csv}")


if __name__ == "__main__":
    main()