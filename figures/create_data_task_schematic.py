from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "data_task_schematic_poster.png"

TEST_NC = ROOT / ".data" / "downscaling_splits" / "test.nc"
TOPO_NC = ROOT / ".data" / "ETOPO2" / "topography_on_gridmet_masked.nc"


def as_celsius(arr):
    """Convert Kelvin-like temperature arrays to Celsius."""
    vals = np.asarray(arr)
    if np.nanmean(vals) > 100:
        vals = vals - 273.15
    return vals


def add_map_frame(ax, title, subtitle):
    ax.set_title("")
    ax.text(
        0.0,
        1.095,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color="#111827",
    )
    ax.text(
        0.0,
        1.045,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color="#4b5563",
    )
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.tick_params(labelsize=9, length=3)
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("#1f2937")


def main():
    ds = xr.open_dataset(TEST_NC)
    topo = xr.open_dataset(TOPO_NC)

    # Choose a warm test-period day with strong spatial structure.
    domain_mean_tmax = ds["tmax_highres"].mean(("lat", "lon")).values
    day_index = int(np.nanargmax(domain_mean_tmax))
    date = np.datetime_as_string(ds["time"].values[day_index], unit="D")

    low = as_celsius(ds["tmax_lowres"].isel(time=day_index).values)
    high = as_celsius(ds["tmax_highres"].isel(time=day_index).values)
    valid = ds["valid_mask"].values.astype(bool)
    elev = np.where(valid, topo["z"].values, np.nan)

    lat_c = ds["lat_coarse"].values
    lon_c = ds["lon_coarse"].values
    lat = ds["lat"].values
    lon = ds["lon"].values

    high = np.where(valid, high, np.nan)

    vmin = np.nanpercentile(np.concatenate([low.ravel(), high.ravel()]), 2)
    vmax = np.nanpercentile(np.concatenate([low.ravel(), high.ravel()]), 98)

    fig = plt.figure(figsize=(16, 7.2), dpi=300)
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[0.16, 1.0],
        width_ratios=[1.0, 0.18, 1.0, 1.0],
        left=0.045,
        right=0.965,
        top=0.90,
        bottom=0.13,
        wspace=0.22,
        hspace=0.26,
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.0,
        0.62,
        "Data and Downscaling Task",
        fontsize=22,
        fontweight="bold",
        color="#111827",
        ha="left",
        va="center",
    )
    title_ax.text(
        0.0,
        0.08,
        "Summer daily maximum 2 m temperature over the northeastern United States",
        fontsize=12,
        color="#4b5563",
        ha="left",
        va="center",
    )

    ax_low = fig.add_subplot(gs[1, 0])
    ax_arrow = fig.add_subplot(gs[1, 1])
    ax_high = fig.add_subplot(gs[1, 2])
    ax_elev = fig.add_subplot(gs[1, 3])

    im0 = ax_low.pcolormesh(lon_c, lat_c, low, cmap="turbo", vmin=vmin, vmax=vmax, shading="auto")
    lon_c_grid, lat_c_grid = np.meshgrid(lon_c, lat_c)
    ax_low.scatter(lon_c_grid, lat_c_grid, s=3, c="black", alpha=0.28, linewidths=0)
    add_map_frame(ax_low, "Input", "ERA5 Tmax, 0.25 deg (~25 km)")
    ax_low.text(
        0.02,
        0.03,
        date,
        transform=ax_low.transAxes,
        fontsize=10,
        color="#111827",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, boxstyle="round,pad=0.25"),
    )

    ax_arrow.axis("off")
    arrow = FancyArrowPatch(
        (0.1, 0.55),
        (0.9, 0.55),
        transform=ax_arrow.transAxes,
        arrowstyle="-|>",
        mutation_scale=24,
        linewidth=2.2,
        color="#1d4ed8",
    )
    ax_arrow.add_patch(arrow)
    ax_arrow.text(
        0.5,
        0.64,
        "model learns",
        ha="center",
        va="bottom",
        fontsize=11,
        color="#1d4ed8",
        fontweight="bold",
        transform=ax_arrow.transAxes,
    )
    ax_arrow.text(
        0.5,
        0.47,
        "coarse-to-fine\nmapping",
        ha="center",
        va="top",
        fontsize=10,
        color="#374151",
        transform=ax_arrow.transAxes,
    )

    im1 = ax_high.pcolormesh(lon, lat, high, cmap="turbo", vmin=vmin, vmax=vmax, shading="auto")
    add_map_frame(ax_high, "Target", "GRIDMET Tmax, 4 km")
    ax_high.text(
        0.02,
        0.03,
        "High-resolution training target",
        transform=ax_high.transAxes,
        fontsize=10,
        color="#111827",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, boxstyle="round,pad=0.25"),
    )

    im2 = ax_elev.pcolormesh(lon, lat, elev, cmap="terrain", shading="auto")
    add_map_frame(ax_elev, "Static Predictor", "Elevation on 4 km grid")
    ax_elev.text(
        0.02,
        0.03,
        "Topography helps explain local Tmax variation",
        transform=ax_elev.transAxes,
        fontsize=10,
        color="#111827",
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, boxstyle="round,pad=0.25"),
    )

    for ax in [ax_low, ax_high, ax_elev]:
        ax.set_xlim(float(lon.min()), float(lon.max()))
        ax.set_ylim(float(lat.min()), float(lat.max()))

    # Small locator rectangle in the input panel to hint that this is a NE U.S. zoom.
    rect = Rectangle(
        (float(lon.min()), float(lat.min())),
        float(lon.max() - lon.min()),
        float(lat.max() - lat.min()),
        fill=False,
        edgecolor="#111827",
        linewidth=1.5,
    )
    ax_low.add_patch(rect)

    cbar0 = fig.colorbar(im1, ax=[ax_low, ax_high], orientation="horizontal", fraction=0.06, pad=0.12)
    cbar0.set_label("Daily maximum 2 m temperature (deg C)", fontsize=11)
    cbar0.ax.tick_params(labelsize=9)

    cbar1 = fig.colorbar(im2, ax=ax_elev, orientation="horizontal", fraction=0.06, pad=0.12)
    cbar1.set_label("Elevation (m)", fontsize=11)
    cbar1.ax.tick_params(labelsize=9)

    fig.text(
        0.045,
        0.035,
        "Training split: 2000-2018 | validation: 2019-2021 | testing: 2022-2025",
        fontsize=12,
        color="#374151",
    )

    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    print(OUT)

    ds.close()
    topo.close()


if __name__ == "__main__":
    main()
