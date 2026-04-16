import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class DownscaleDatasetTElevUrban(Dataset):
    def __init__(self, dynamic_nc_path, topo_nc_path, urban_nc_path):
        ds_dyn = xr.open_dataset(dynamic_nc_path)
        ds_topo = xr.open_dataset(topo_nc_path)
        ds_urban = xr.open_dataset(urban_nc_path)

        required_dyn = ["x_norm", "y_norm", "valid_mask"]
        for v in required_dyn:
            if v not in ds_dyn:
                raise ValueError(f"Missing dynamic variable '{v}' in {dynamic_nc_path}")

        if "elevation" not in ds_topo:
            raise ValueError(f"Missing topo variable 'elevation' in {topo_nc_path}")

        if "urban_fraction_norm" not in ds_urban:
            raise ValueError(f"Missing urban variable 'urban_fraction_norm' in {urban_nc_path}")

        # coordinate checks
        dyn_lat = ds_dyn["lat"].values
        dyn_lon = ds_dyn["lon"].values
        topo_lat = ds_topo["lat"].values
        topo_lon = ds_topo["lon"].values
        urban_lat = ds_urban["lat"].values
        urban_lon = ds_urban["lon"].values

        for other_lat, other_lon, name in [
            (topo_lat, topo_lon, "topo"),
            (urban_lat, urban_lon, "urban"),
        ]:
            if dyn_lat.shape != other_lat.shape or dyn_lon.shape != other_lon.shape:
                raise ValueError(f"Dynamic file and {name} file do not have matching lat/lon shapes.")
            if not np.allclose(dyn_lat, other_lat):
                raise ValueError(f"Latitude coordinates do not match between dynamic and {name} files.")
            if not np.allclose(dyn_lon, other_lon):
                raise ValueError(f"Longitude coordinates do not match between dynamic and {name} files.")

        x = ds_dyn["x_norm"].values.astype(np.float32)                 # (time, lat, lon)
        y = ds_dyn["y_norm"].values.astype(np.float32)                 # (time, lat, lon)
        mask = ds_dyn["valid_mask"].values.astype(np.float32)          # (lat, lon)
        elevation = ds_topo["elevation"].values.astype(np.float32)     # (lat, lon)
        urban = ds_urban["urban_fraction_norm"].values.astype(np.float32)  # (lat, lon)

        ds_dyn.close()
        ds_topo.close()
        ds_urban.close()

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        elevation = np.nan_to_num(elevation, nan=0.0, posinf=0.0, neginf=0.0)
        urban = np.nan_to_num(urban, nan=0.0, posinf=0.0, neginf=0.0)

        x = x[:, None, :, :]                  # (time, 1, lat, lon)
        y = y[:, None, :, :]                  # (time, 1, lat, lon)
        elevation = elevation[None, :, :]     # (1, lat, lon)
        urban = urban[None, :, :]             # (1, lat, lon)
        mask = mask[None, :, :]               # (1, lat, lon)

        elevation_tiled = np.broadcast_to(
            elevation, (x.shape[0], 1, x.shape[2], x.shape[3])
        )
        urban_tiled = np.broadcast_to(
            urban, (x.shape[0], 1, x.shape[2], x.shape[3])
        )

        self.x = np.concatenate([x, elevation_tiled, urban_tiled], axis=1).astype(np.float32)
        self.y = y.astype(np.float32)
        self.mask = mask.astype(np.float32)

        self.n_samples = self.x.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),   # (3, H, W)
            torch.from_numpy(self.y[idx]),   # (1, H, W)
            torch.from_numpy(self.mask),     # (1, H, W)
        )