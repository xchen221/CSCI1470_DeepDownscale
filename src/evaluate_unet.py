import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from dataset_window import DownscaleWindowDataset
from model_unet import UNet


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


def masked_bias(pred, target, mask):
    valid = mask > 0.5
    return (pred[valid] - target[valid]).mean()


def main():
    test_norm_path = ".data/downscaling_splits/test_norm.nc"
    norm_stats_path = ".data/downscaling_splits/norm_stats.nc"
    checkpoint_path = "outputs/checkpoints/best_unet_w3_elev_date.pt"

    window_size = 3
    use_elev = True
    use_date = True
    in_channels = window_size + int(use_elev) + 2 * int(use_date)

    device = get_device()
    print("Using device:", device)

    # model input dataset
    test_ds_window = DownscaleWindowDataset(
        test_norm_path,
        window_size=window_size,
        use_elev=use_elev,
        use_date=use_date,
    )
    test_loader = DataLoader(test_ds_window, batch_size=4, shuffle=False, num_workers=0)

    # full xarray test set
    ds_test = xr.open_dataset(test_norm_path)
    ds_test_aligned = ds_test.isel(time=slice(window_size - 1, None))

    stats = xr.open_dataset(norm_stats_path)
    y_mean = float(stats["y_mean"].values)
    y_std = float(stats["y_std"].values)

    model = UNet(in_channels=in_channels, out_channels=1, base_ch=32).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    preds_norm = []

    with torch.no_grad():
        for xb, yb, mb in test_loader:
            xb = xb.to(device)
            pred = model(xb).cpu().numpy()   # [B, 1, H, W]
            preds_norm.append(pred)

    preds_norm = np.concatenate(preds_norm, axis=0)[:, 0, :, :]   # [time, H, W]

    # denormalize residual
    pred_residual = preds_norm * y_std + y_mean

    truth_high = ds_test_aligned["tmax_highres"].values
    low_interp = ds_test_aligned["tmax_lowres_interp"].values
    mask = ds_test_aligned["valid_mask"].values.astype(bool)

    pred_high = low_interp + pred_residual

    mask_3d = np.broadcast_to(mask, truth_high.shape)

    baseline_mae = masked_mae(low_interp, truth_high, mask_3d)
    baseline_rmse = masked_rmse(low_interp, truth_high, mask_3d)
    baseline_bias = masked_bias(low_interp, truth_high, mask_3d)

    unet_mae = masked_mae(pred_high, truth_high, mask_3d)
    unet_rmse = masked_rmse(pred_high, truth_high, mask_3d)
    unet_bias = masked_bias(pred_high, truth_high, mask_3d)

    print("\nTest metrics in physical units:")
    print(f"Baseline interp MAE : {baseline_mae:.4f}")
    print(f"Baseline interp RMSE: {baseline_rmse:.4f}")
    print(f"Baseline interp Bias: {baseline_bias:.4f}")
    print(f"U-Net pred MAE      : {unet_mae:.4f}")
    print(f"U-Net pred RMSE     : {unet_rmse:.4f}")
    print(f"U-Net pred Bias     : {unet_bias:.4f}")

    ds_pred = xr.Dataset(
        data_vars={
            "pred_residual": (("time", "lat", "lon"), pred_residual.astype(np.float32)),
            "pred_tmax_highres": (("time", "lat", "lon"), pred_high.astype(np.float32)),
        },
        coords={
            "time": ds_test_aligned["time"].values,
            "lat": ds_test_aligned["lat"].values,
            "lon": ds_test_aligned["lon"].values,
        },
        attrs={"description": "U-Net predictions on test set"},
    )
    ds_pred.to_netcdf("outputs/test_predictions_unet_w3_elev_date.nc")
    print("\nSaved predictions to outputs/test_predictions_unet_w3_elev_date.nc")


if __name__ == "__main__":
    main()