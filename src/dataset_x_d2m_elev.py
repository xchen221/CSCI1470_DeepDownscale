import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class DownscaleDatasetXD2MElev(Dataset):
    def __init__(self, base_nc_path, extra_nc_path, topo_nc_path):
        ds_base = xr.open_dataset(base_nc_path)
        ds_extra = xr.open_dataset(extra_nc_path)
        ds_topo = xr.open_dataset(topo_nc_path)

        required_base = ["x_norm", "y_norm", "valid_mask"]
        for v in required_base:
            if v not in ds_base:
                raise ValueError(f"Missing base variable '{v}' in {base_nc_path}")

        if "d2m_norm" not in ds_extra:
            raise ValueError(f"Missing extra variable 'd2m_norm' in {extra_nc_path}")

        if "elevation" not in ds_topo:
            raise ValueError(f"Missing topo variable 'elevation' in {topo_nc_path}")

        base_lat = ds_base["lat"].values
        base_lon = ds_base["lon"].values
        extra_lat = ds_extra["lat"].values
        extra_lon = ds_extra["lon"].values
        topo_lat = ds_topo["lat"].values
        topo_lon = ds_topo["lon"].values

        if not np.allclose(base_lat, extra_lat):
            raise ValueError("Latitude coordinates do not match between base and extra files.")
        if not np.allclose(base_lon, extra_lon):
            raise ValueError("Longitude coordinates do not match between base and extra files.")
        if not np.allclose(base_lat, topo_lat):
            raise ValueError("Latitude coordinates do not match between base and topo files.")
        if not np.allclose(base_lon, topo_lon):
            raise ValueError("Longitude coordinates do not match between base and topo files.")

        base_time = ds_base["time"].values
        extra_time = ds_extra["time"].values
        if not np.array_equal(base_time, extra_time):
            raise ValueError("Time coordinates do not match between base and extra files.")

        x = ds_base["x_norm"].values.astype(np.float32)
        y = ds_base["y_norm"].values.astype(np.float32)
        mask = ds_base["valid_mask"].values.astype(np.float32)

        d2m = ds_extra["d2m_norm"].values.astype(np.float32)
        elevation = ds_topo["elevation"].values.astype(np.float32)

        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
        d2m = np.nan_to_num(d2m, nan=0.0, posinf=0.0, neginf=0.0)
        elevation = np.nan_to_num(elevation, nan=0.0, posinf=0.0, neginf=0.0)

        elevation = elevation[None, :, :]
        mask = mask[None, :, :]

        ntime, nlat, nlon = x.shape
        elevation_tiled = np.broadcast_to(elevation, (ntime, 1, nlat, nlon))

        x = x[:, None, :, :]
        d2m = d2m[:, None, :, :]

        self.x = np.concatenate([x, d2m, elevation_tiled], axis=1).astype(np.float32)
        self.y = y[:, None, :, :].astype(np.float32)
        self.mask = mask.astype(np.float32)

        self.n_samples = self.x.shape[0]

        ds_base.close()
        ds_extra.close()
        ds_topo.close()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]),
            torch.from_numpy(self.mask),
        )