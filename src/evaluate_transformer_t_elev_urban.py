import os

import numpy as np
import torch
import xarray as xr
from torch.utils.data import DataLoader

from dataset_t_elev_urban import DownscaleDatasetTElevUrban
from model_transformer import UNetTransformerDownscale


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def main():
    test_path = ".data/downscaling_splits/test_norm.nc"
    topo_path = ".data/ETOPO2/topography_features_on_gridmet_masked_norm.nc"
    urban_path = ".data/Zenodo/urban_fraction_on_gridmet_masked_norm.nc"
    stats_path = ".data/downscaling_splits/norm_stats.nc"

    checkpoint_path = "outputs/checkpoints/best_transformer_t_elev_urban_oscar.pt"
    output_path = "outputs/test_predictions_transformer_t_elev_urban_oscar.nc"

    batch_size = 4

    device = get_device()
    print("Using device:", device)

    # dataset / loader
    test_ds = DownscaleDatasetTElevUrban(test_path, topo_path, urban_path)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )

    # model: must match training config exactly
    model = (
        UNetTransformerDownscale(
            in_channels=3,
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

    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    preds = []

    with torch.no_grad():
        for x, _, _ in test_loader:
            x = x.float().to(device).contiguous()
            pred = model(x)
            preds.append(pred.cpu().numpy())

    preds = np.concatenate(preds, axis=0)  # (time, 1, lat, lon)
    preds = preds[:, 0, :, :]  # (time, lat, lon)

    # recover metadata and denormalize
    ds_test = xr.open_dataset(test_path)
    ds_stats = xr.open_dataset(stats_path)

    y_mean = float(ds_stats["y_mean"].values)
    y_std = float(ds_stats["y_std"].values)

    # predicted normalized residual -> physical residual
    pred_residual = preds * y_std + y_mean

    # reconstruct high-res Tmax prediction
    pred_tmax_highres = ds_test["tmax_lowres_interp"].values + pred_residual

    ds_out = xr.Dataset(
        {
            "pred_residual": (
                ("time", "lat", "lon"),
                pred_residual.astype(np.float32),
            ),
            "pred_tmax_highres": (
                ("time", "lat", "lon"),
                pred_tmax_highres.astype(np.float32),
            ),
        },
        coords={
            "time": ds_test["time"].values,
            "lat": ds_test["lat"].values,
            "lon": ds_test["lon"].values,
        },
        attrs={
            "description": "Transformer U-Net bottleneck (temperature + elevation + urban) predictions on test set"
        },
    )

    os.makedirs("outputs", exist_ok=True)
    ds_out.to_netcdf(output_path)

    ds_test.close()
    ds_stats.close()

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()