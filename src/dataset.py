# src/dataset.py

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class DownscaleDataset(Dataset):
    def __init__(self, nc_path):
        self.ds = xr.open_dataset(nc_path)

        required_vars = ["x_norm", "z_norm", "y_norm", "valid_mask"]
        for v in required_vars:
            if v not in self.ds:
                raise ValueError(f"Missing variable '{v}' in {nc_path}")

        # static fields: load once
        self.z_static = self.ds["z_norm"].values.astype(np.float32)          # (lat, lon)
        self.mask = self.ds["valid_mask"].values.astype(np.float32)          # (lat, lon)

        self.n_samples = self.ds.sizes["time"]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # dynamic input for one day
        x_dynamic = self.ds["x_norm"].isel(time=idx).values.astype(np.float32)
        y = self.ds["y_norm"].isel(time=idx).values.astype(np.float32)

        x_dynamic = np.nan_to_num(x_dynamic, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        z_static = np.nan_to_num(self.z_static, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(self.mask, nan=0.0, posinf=0.0, neginf=0.0)

        x = np.stack([x_dynamic, z_static], axis=0)
        y = np.expand_dims(y, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask),
        )