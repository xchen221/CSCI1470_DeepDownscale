import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class DownscaleDatasetRich(Dataset):
    def __init__(self, base_nc_path, extra_nc_path, topo_nc_path):
        ds_base = xr.open_dataset(base_nc_path)
        ds_extra = xr.open_dataset(extra_nc_path)
        ds_topo = xr.open_dataset(topo_nc_path)

        # -------------------------
        # required variables
        # -------------------------
        required_base = ["x_norm", "y_norm", "valid_mask"]
        for v in required_base:
            if v not in ds_base:
                raise ValueError(f"Missing base variable '{v}' in {base_nc_path}")

        required_extra = ["ssrd_norm", "d2m_norm"]
        for v in required_extra:
            if v not in ds_extra:
                raise ValueError(f"Missing extra variable '{v}' in {extra_nc_path}")

        required_topo = [
            "elevation_mean",
            "elevation_std",
            "slope_mean",
            "slope_std",
        ]
        for v in required_topo:
            if v not in ds_topo:
                raise ValueError(f"Missing topo variable '{v}' in {topo_nc_path}")

        # -------------------------
        # coordinate checks
        # -------------------------
        base_lat = ds_base["lat"].values
        base_lon = ds_base["lon"].values

        extra_lat = ds_extra["lat"].values
        extra_lon = ds_extra["lon"].values

        topo_lat = ds_topo["lat"].values
        topo_lon = ds_topo["lon"].values

        if base_lat.shape != extra_lat.shape or base_lon.shape != extra_lon.shape:
            raise ValueError("Base file and extra file do not have matching lat/lon shapes.")
        if base_lat.shape != topo_lat.shape or base_lon.shape != topo_lon.shape:
            raise ValueError("Base file and topo file do not have matching lat/lon shapes.")

        if not np.allclose(base_lat, extra_lat):
            raise ValueError("Latitude coordinates do not match between base and extra files.")
        if not np.allclose(base_lon, extra_lon):
            raise ValueError("Longitude coordinates do not match between base and extra files.")

        if not np.allclose(base_lat, topo_lat):
            raise ValueError("Latitude coordinates do not match between base and topo files.")
        if not np.allclose(base_lon, topo_lon):
            raise ValueError("Longitude coordinates do not match between base and topo files.")

        # -------------------------
        # time checks
        # -------------------------
        base_time = ds_base["time"].values
        extra_time = ds_extra["time"].values

        if base_time.shape != extra_time.shape:
            raise ValueError("Base file and extra file do not have matching time shapes.")
        if not np.array_equal(base_time, extra_time):
            raise ValueError("Time coordinates do not match between base and extra files.")

        # -------------------------
        # load arrays once
        # -------------------------
        x = ds_base["x_norm"].values.astype(np.float32)            # (time, lat, lon)
        y = ds_base["y_norm"].values.astype(np.float32)            # (time, lat, lon)
        mask = ds_base["valid_mask"].values.astype(np.float32)     # (lat, lon)

        ssrd = ds_extra["ssrd_norm"].values.astype(np.float32)     # (time, lat, lon)
        d2m = ds_extra["d2m_norm"].values.astype(np.float32)       # (time, lat, lon)

        elevation_mean = ds_topo["elevation_mean"].values.astype(np.float32)  # (lat, lon)
        elevation_std = ds_topo["elevation_std"].values.astype(np.float32)    # (lat, lon)
        slope_mean = ds_topo["slope_mean"].values.astype(np.float32)          # (lat, lon)
        slope_std = ds_topo["slope_std"].values.astype(np.float32)            # (lat, lon)

        # -------------------------
        # clean once
        # -------------------------
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

        ssrd = np.nan_to_num(ssrd, nan=0.0, posinf=0.0, neginf=0.0)
        d2m = np.nan_to_num(d2m, nan=0.0, posinf=0.0, neginf=0.0)

        elevation_mean = np.nan_to_num(elevation_mean, nan=0.0, posinf=0.0, neginf=0.0)
        elevation_std = np.nan_to_num(elevation_std, nan=0.0, posinf=0.0, neginf=0.0)
        slope_mean = np.nan_to_num(slope_mean, nan=0.0, posinf=0.0, neginf=0.0)
        slope_std = np.nan_to_num(slope_std, nan=0.0, posinf=0.0, neginf=0.0)

        # -------------------------
        # expand static fields once
        # -------------------------
        elevation_mean = elevation_mean[None, :, :]   # (1, lat, lon)
        elevation_std = elevation_std[None, :, :]
        slope_mean = slope_mean[None, :, :]
        slope_std = slope_std[None, :, :]
        mask = mask[None, :, :]                       # (1, lat, lon)

        ntime, nlat, nlon = x.shape

        elevation_mean_tiled = np.broadcast_to(elevation_mean, (ntime, 1, nlat, nlon))
        elevation_std_tiled = np.broadcast_to(elevation_std, (ntime, 1, nlat, nlon))
        slope_mean_tiled = np.broadcast_to(slope_mean, (ntime, 1, nlat, nlon))
        slope_std_tiled = np.broadcast_to(slope_std, (ntime, 1, nlat, nlon))

        # -------------------------
        # add channel dimension to dynamic fields
        # -------------------------
        x = x[:, None, :, :]        # (time, 1, lat, lon)
        ssrd = ssrd[:, None, :, :]
        d2m = d2m[:, None, :, :]

        # -------------------------
        # build full input once
        # channels:
        # 0 = x_norm
        # 1 = ssrd_norm
        # 2 = d2m_norm
        # 3 = elevation_mean
        # 4 = elevation_std
        # 5 = slope_mean
        # 6 = slope_std
        # -------------------------
        self.x = np.concatenate(
            [
                x,
                ssrd,
                d2m,
                elevation_mean_tiled,
                elevation_std_tiled,
                slope_mean_tiled,
                slope_std_tiled,
            ],
            axis=1,
        ).astype(np.float32)

        self.y = y[:, None, :, :].astype(np.float32)   # (time, 1, lat, lon)
        self.mask = mask.astype(np.float32)            # (1, lat, lon)

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