import numpy as np
import torch
import xarray as xr
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
    test_norm_path = ".data/downscaling_splits/test_norm.nc"
    norm_stats_path = ".data/downscaling_splits/norm_stats.nc"
    checkpoint_path = "outputs/checkpoints/best_cnn.pt"

    device = get_device()
    print("Using device:", device)

    # load normalized test set for model input
    test_ds = DownscaleDataset(test_norm_path)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)

    # load original xarray dataset too, for physical fields
    ds_test = xr.open_dataset(test_norm_path)
    stats = xr.open_dataset(norm_stats_path)

    y_mean = float(stats["y_mean"].values)
    y_std = float(stats["y_std"].values)

    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preds_norm = []

    with torch.no_grad():
        for xb, yb, mb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()   # [B, 1, H, W]
            preds_norm.append(pred)

    preds_norm = np.concatenate(preds_norm, axis=0)[:, 0, :, :]   # [time, H, W]

    # denormalize residual prediction
    pred_residual = preds_norm * y_std + y_mean

    # ground truth and baseline in physical units
    truth_high = ds_test["tmax_highres"].values
    low_interp = ds_test["tmax_lowres_interp"].values
    mask = ds_test["valid_mask"].values.astype(bool)

    # reconstruct predicted high-res Tmax
    pred_high = low_interp + pred_residual

    # expand mask to time dimension
    mask_3d = np.broadcast_to(mask, truth_high.shape)

    # metrics
    baseline_mae = masked_mae(low_interp, truth_high, mask_3d)
    baseline_rmse = masked_rmse(low_interp, truth_high, mask_3d)

    cnn_mae = masked_mae(pred_high, truth_high, mask_3d)
    cnn_rmse = masked_rmse(pred_high, truth_high, mask_3d)

    print("\nTest metrics in physical units:")
    print(f"Baseline interp MAE : {baseline_mae:.4f}")
    print(f"Baseline interp RMSE: {baseline_rmse:.4f}")
    print(f"CNN pred MAE        : {cnn_mae:.4f}")
    print(f"CNN pred RMSE       : {cnn_rmse:.4f}")

    # save predictions
    ds_pred = xr.Dataset(
        data_vars={
            "pred_residual": (("time", "lat", "lon"), pred_residual.astype(np.float32)),
            "pred_tmax_highres": (("time", "lat", "lon"), pred_high.astype(np.float32)),
        },
        coords={
            "time": ds_test["time"].values,
            "lat": ds_test["lat"].values,
            "lon": ds_test["lon"].values,
        },
        attrs={
            "description": "CNN predictions on test set"
        }
    )
    ds_pred.to_netcdf("outputs/test_predictions.nc")
    print("\nSaved predictions to outputs/test_predictions.nc")


if __name__ == "__main__":
    main()