import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class DownscaleDatasetResUnet(Dataset):
    def __init__(self, dynamic_nc_path, topo_nc_path):
        ds_dyn = xr.open_dataset(dynamic_nc_path)
        ds_topo = xr.open_dataset(topo_nc_path)

        required_dyn = ["x_norm", "y_norm", "valid_mask"]
        for v in required_dyn:
            if v not in ds_dyn:
                raise ValueError(f"Missing dynamic variable '{v}' in {dynamic_nc_path}")

        if "elevation" not in ds_topo:
            raise ValueError(f"Missing topo variable 'elevation' in {topo_nc_path}")

        dyn_lat = ds_dyn["lat"].values
        dyn_lon = ds_dyn["lon"].values
        topo_lat = ds_topo["lat"].values
        topo_lon = ds_topo["lon"].values

        if dyn_lat.shape != topo_lat.shape or dyn_lon.shape != topo_lon.shape:
            raise ValueError("Dynamic file and topo file do not have matching lat/lon shapes.")
        if not np.allclose(dyn_lat, topo_lat):
            raise ValueError("Latitude coordinates do not match between dynamic and topo files.")
        if not np.allclose(dyn_lon, topo_lon):
            raise ValueError("Longitude coordinates do not match between dynamic and topo files.")

        x = ds_dyn["x_norm"].values.astype(np.float32)
        y = ds_dyn["y_norm"].values.astype(np.float32)
        mask = ds_dyn["valid_mask"].values.astype(np.float32)
        elevation = ds_topo["elevation"].values.astype(np.float32)

        ds_dyn.close()
        ds_topo.close()

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        elevation = np.nan_to_num(elevation, nan=0.0, posinf=0.0, neginf=0.0)

        x = x[:, None, :, :]
        y = y[:, None, :, :]
        elevation = elevation[None, :, :]
        mask = mask[None, :, :]

        elevation_tiled = np.broadcast_to(
            elevation, (x.shape[0], 1, x.shape[2], x.shape[3])
        )

        self.x = np.concatenate([x, elevation_tiled], axis=1).astype(np.float32)
        self.y = y.astype(np.float32)
        self.mask = mask.astype(np.float32)

        self.n_samples = self.x.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]),
            torch.from_numpy(self.mask),
        )