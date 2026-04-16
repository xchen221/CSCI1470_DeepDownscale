import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class DownscaleDatasetXOnly(Dataset):
    def __init__(self, dynamic_nc_path):
        ds_dyn = xr.open_dataset(dynamic_nc_path)

        required_dyn = ["x_norm", "y_norm", "valid_mask"]
        for v in required_dyn:
            if v not in ds_dyn:
                raise ValueError(f"Missing dynamic variable '{v}' in {dynamic_nc_path}")

        # Load once
        x = ds_dyn["x_norm"].values.astype(np.float32)         # (time, lat, lon)
        y = ds_dyn["y_norm"].values.astype(np.float32)         # (time, lat, lon)
        mask = ds_dyn["valid_mask"].values.astype(np.float32)  # (lat, lon)

        # Clean once
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)

        # Expand once
        x = x[:, None, :, :]        # (time, 1, lat, lon)
        y = y[:, None, :, :]        # (time, 1, lat, lon)
        mask = mask[None, :, :]     # (1, lat, lon)

        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
        self.mask = mask.astype(np.float32)

        self.n_samples = self.x.shape[0]

        ds_dyn.close()

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]),
            torch.from_numpy(self.y[idx]),
            torch.from_numpy(self.mask),
        )