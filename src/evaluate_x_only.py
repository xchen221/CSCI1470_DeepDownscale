import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from dataset_x_only import DownscaleDatasetXOnly
from model_cnn import UNetDownscale


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    test_path = ".data/downscaling_splits/test_norm.nc"
    checkpoint_path = "outputs/checkpoints/best_unet_x_only.pt"
    output_path = "outputs/test_predictions_x_only.nc"

    batch_size = 4
    num_workers = 0

    device = get_device()
    print("Using device:", device)

    # dataset / loader
    test_ds = DownscaleDatasetXOnly(test_path)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # model
    model = UNetDownscale(in_channels=1).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preds = []

    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.to(device)
            pred = model(x)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)   # (time, 1, lat, lon)
    preds = preds[:, 0, :, :]               # (time, lat, lon)

    # read test file to recover metadata and denormalize
    ds_test = xr.open_dataset(test_path)
    ds_stats = xr.open_dataset(".data/downscaling_splits/norm_stats.nc")

    y_mean = float(ds_stats["y_mean"].values)
    y_std = float(ds_stats["y_std"].values)

    # prediction is normalized residual
    pred_residual = preds * y_std + y_mean

    # reconstruct high-res Tmax prediction
    pred_tmax_highres = ds_test["tmax_lowres_interp"].values + pred_residual

    ds_out = xr.Dataset(
        {
            "pred_residual": (("time", "lat", "lon"), pred_residual.astype(np.float32)),
            "pred_tmax_highres": (("time", "lat", "lon"), pred_tmax_highres.astype(np.float32)),
        },
        coords={
            "time": ds_test["time"].values,
            "lat": ds_test["lat"].values,
            "lon": ds_test["lon"].values,
        },
        attrs={
            "description": "U-Net predictions on test set using x_norm only"
        }
    )

    os.makedirs("outputs", exist_ok=True)

    if os.path.exists(output_path):
        os.remove(output_path)

    ds_out.to_netcdf(output_path)
    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()