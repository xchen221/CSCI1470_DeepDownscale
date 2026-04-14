import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class DownscaleWindowDataset(Dataset):
    def __init__(self, nc_path, window_size=3, use_elev=True, use_date=True):
        self.ds = xr.open_dataset(nc_path)

        self.window_size = window_size
        self.use_elev = use_elev
        self.use_date = use_date

        required_vars = ["x_norm", "y_norm", "valid_mask"]
        for v in required_vars:
            if v not in self.ds:
                raise ValueError(f"Missing variable '{v}' in {nc_path}")

        if self.use_elev and "z_norm" not in self.ds:
            raise ValueError(f"Missing variable 'z_norm' in {nc_path}")

        # Load arrays
        self.x = self.ds["x_norm"].values.astype(np.float32)         # (time, lat, lon)
        self.y = self.ds["y_norm"].values.astype(np.float32)         # (time, lat, lon)
        self.mask = self.ds["valid_mask"].values.astype(np.float32)  # (lat, lon)

        self.z = None
        if self.use_elev:
            self.z = self.ds["z_norm"].values.astype(np.float32)     # (lat, lon)

        # Day-of-year for each target day
        self.time = self.ds["time"].values
        self.doy = xr.DataArray(self.time).dt.dayofyear.values.astype(np.float32)

        self.n_time = self.ds.sizes["time"]
        self.n_samples = self.n_time - self.window_size + 1

        if self.n_samples <= 0:
            raise ValueError(
                f"window_size={window_size} is larger than available time steps={self.n_time}"
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # target day index
        t = idx + self.window_size - 1

        # window of low-res input ending at target day t
        x_window = self.x[idx:t + 1]   # shape: (window_size, lat, lon)

        # target residual on day t
        y_target = self.y[t]           # shape: (lat, lon)

        # safety: remove any leftover nan/inf
        x_window = np.nan_to_num(x_window, nan=0.0, posinf=0.0, neginf=0.0)
        y_target = np.nan_to_num(y_target, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(self.mask, nan=0.0, posinf=0.0, neginf=0.0)

        channels = []

        # 3-day low-res window
        for i in range(self.window_size):
            channels.append(x_window[i])

        # elevation
        if self.use_elev:
            z = np.nan_to_num(self.z, nan=0.0, posinf=0.0, neginf=0.0)
            channels.append(z)

        # sin/cos day-of-year of target day
        if self.use_date:
            doy_t = self.doy[t]
            sin_doy = np.sin(2.0 * np.pi * doy_t / 365.0).astype(np.float32)
            cos_doy = np.cos(2.0 * np.pi * doy_t / 365.0).astype(np.float32)

            h, w = y_target.shape
            sin_map = np.full((h, w), sin_doy, dtype=np.float32)
            cos_map = np.full((h, w), cos_doy, dtype=np.float32)

            channels.append(sin_map)
            channels.append(cos_map)

        x_out = np.stack(channels, axis=0).astype(np.float32)   # (C, lat, lon)
        y_out = np.expand_dims(y_target, axis=0).astype(np.float32)   # (1, lat, lon)
        mask_out = np.expand_dims(mask, axis=0).astype(np.float32)    # (1, lat, lon)

        return (
            torch.from_numpy(x_out),
            torch.from_numpy(y_out),
            torch.from_numpy(mask_out),
        )