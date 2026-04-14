# src/dataset.py

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset


class DownscaleDataset(Dataset):
    def __init__(self, dynamic_nc_path, topo_nc_path):
        self.ds_dyn = xr.open_dataset(dynamic_nc_path)
        self.ds_topo = xr.open_dataset(topo_nc_path)

        required_dyn = ["x_norm", "y_norm", "valid_mask"]
        for v in required_dyn:
            if v not in self.ds_dyn:
                raise ValueError(f"Missing dynamic variable '{v}' in {dynamic_nc_path}")

        if "elevation" not in self.ds_topo:
            raise ValueError(f"Missing topo variable 'elevation' in {topo_nc_path}")

        dyn_lat = self.ds_dyn["lat"].values
        dyn_lon = self.ds_dyn["lon"].values
        topo_lat = self.ds_topo["lat"].values
        topo_lon = self.ds_topo["lon"].values

        if dyn_lat.shape != topo_lat.shape or dyn_lon.shape != topo_lon.shape:
            raise ValueError("Dynamic file and topo file do not have matching lat/lon shapes.")

        if not np.allclose(dyn_lat, topo_lat):
            raise ValueError("Latitude coordinates do not match between dynamic and topo files.")

        if not np.allclose(dyn_lon, topo_lon):
            raise ValueError("Longitude coordinates do not match between dynamic and topo files.")

        self.x = self.ds_dyn["x_norm"].values.astype(np.float32)
        self.y = self.ds_dyn["y_norm"].values.astype(np.float32)
        self.mask = self.ds_dyn["valid_mask"].values.astype(np.float32)

        self.elevation = self.ds_topo["elevation"].values.astype(np.float32)

        self.n_samples = self.ds_dyn.sizes["time"]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x_dynamic = self.x[idx]
        y_target = self.y[idx]

        x_dynamic = np.nan_to_num(x_dynamic, nan=0.0, posinf=0.0, neginf=0.0)
        y_target = np.nan_to_num(y_target, nan=0.0, posinf=0.0, neginf=0.0)
        elevation = np.nan_to_num(self.elevation, nan=0.0, posinf=0.0, neginf=0.0)
        mask = np.nan_to_num(self.mask, nan=0.0, posinf=0.0, neginf=0.0)

        x = np.stack([x_dynamic, elevation], axis=0)
        y = np.expand_dims(y_target, axis=0)
        mask = np.expand_dims(mask, axis=0)

        return (
            torch.from_numpy(x),
            torch.from_numpy(y),
            torch.from_numpy(mask),
        )