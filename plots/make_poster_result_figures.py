from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "figures" / "poster_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_PATH = ROOT / ".data" / "downscaling_splits" / "test_norm.nc"
EXTRA_PATH = ROOT / ".data" / "downscaling_splits_extra" / "test_extra_norm.nc"
TOPO_PATH = ROOT / ".data" / "ETOPO2" / "topography_features_on_gridmet_masked_norm.nc"
WEATHER_DAILY_PATH = (
    ROOT
    / "figures"
    / "weather_condition_compare_era5"
    / "daily_metrics_with_era5_weather.csv"
)

PRED_VAR = "pred_tmax_highres"
MODELS = {
    "Interpolation": None,
    "U-Net T+Elev": ROOT / "outputs" / "test_predictions.nc",
    "Transformer T+Elev": ROOT / "outputs" / "test_predictions_transformer_t_elev_oscar.nc",
    "Transformer T+Dew Point+Elev": ROOT
    / "outputs"
    / "test_predictions_transformer_t_elev_d2m.nc",
}

COLORS = {
    "Interpolation": "#777777",
    "U-Net T+Elev": "#2A7AB9",
    "Transformer T+Elev": "#D07A28",
    "Transformer T+Dew Point+Elev": "#2E8B57",
}

RMSE_CMAP = "YlOrRd"
BENEFIT_CMAP = "RdBu"
TEMP_CMAP = "turbo"


plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 140,
        "savefig.dpi": 350,
        "font.family": "DejaVu Sans",
    }
)


def maybe_celsius(da, mask=None):
    if mask is not None and {"lat", "lon"}.issubset(da.dims):
        vals = da.where(mask).values
    else:
        vals = da.values
    return da - 273.15 if float(np.nanmean(vals)) > 100 else da


def spatial_rmse(pred, truth, mask):
    return np.sqrt(((pred - truth) ** 2).where(mask).mean(dim="time"))


def spatial_bias(pred, truth, mask):
    return (pred - truth).where(mask).mean(dim="time")


def daily_mean(da, mask):
    return np.nanmean(da.where(mask).values, axis=(1, 2))


def daily_rmse(pred, truth, mask):
    return np.sqrt(daily_mean((pred - truth) ** 2, mask))


def daily_mae(pred, truth, mask):
    return daily_mean(np.abs(pred - truth), mask)


def daily_bias(pred, truth, mask):
    return daily_mean(pred - truth, mask)


def overall_metrics(pred, truth, mask):
    valid = mask.values.astype(bool)
    pred_values = pred.values
    truth_values = truth.values
    valid3d = np.broadcast_to(valid, truth_values.shape)
    valid3d = valid3d & np.isfinite(pred_values) & np.isfinite(truth_values)
    err = pred_values[valid3d] - truth_values[valid3d]
    return {
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "Bias": float(np.mean(err)),
        "Corr.": float(np.corrcoef(pred_values[valid3d], truth_values[valid3d])[0, 1]),
    }


def add_map(ax, da, title, cmap, vmin=None, vmax=None, cbar_label="deg C"):
    arr = da.values
    finite = np.isfinite(arr)
    if finite.any():
        rows = np.where(finite.any(axis=1))[0]
        cols = np.where(finite.any(axis=0))[0]
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        arr = arr[r0:r1, c0:c1]
        lat = da.lat.values[r0:r1]
        lon = da.lon.values[c0:c1]
    else:
        lat = da.lat.values
        lon = da.lon.values
    im = ax.pcolormesh(
        lon,
        lat,
        arr,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("auto")
    ax.set_title(title, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label(cbar_label)
    return im


def add_map_no_cbar(ax, da, title, cmap, vmin=None, vmax=None):
    arr = da.values
    finite = np.isfinite(arr)
    if finite.any():
        rows = np.where(finite.any(axis=1))[0]
        cols = np.where(finite.any(axis=0))[0]
        r0, r1 = rows[0], rows[-1] + 1
        c0, c1 = cols[0], cols[-1] + 1
        arr = arr[r0:r1, c0:c1]
        lat = da.lat.values[r0:r1]
        lon = da.lon.values[c0:c1]
    else:
        lat = da.lat.values
        lon = da.lon.values
    im = ax.pcolormesh(
        lon,
        lat,
        arr,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_aspect("auto")
    ax.set_title(title, weight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    return im


def quantile_regime(values, labels):
    q1, q2 = np.nanquantile(values, [1 / 3, 2 / 3])
    return pd.cut(values, [-np.inf, q1, q2, np.inf], labels=labels)


def load_data():
    ds_test = xr.open_dataset(TEST_PATH)
    ds_extra = xr.open_dataset(EXTRA_PATH)

    common_time = ds_test.time.values
    for path in MODELS.values():
        if path is not None:
            with xr.open_dataset(path) as ds:
                common_time = np.intersect1d(common_time, ds.time.values)
    common_time = np.intersect1d(common_time, ds_extra.time.values)

    mask = ds_test["valid_mask"].astype(bool)
    truth = maybe_celsius(ds_test["tmax_highres"].sel(time=common_time), mask)
    interp = maybe_celsius(ds_test["tmax_lowres_interp"].sel(time=common_time), mask)
    d2m = maybe_celsius(ds_extra["d2m_lowres_interp"].sel(time=common_time), mask)

    preds = {"Interpolation": interp}
    for name, path in MODELS.items():
        if path is None:
            continue
        with xr.open_dataset(path) as ds:
            preds[name] = maybe_celsius(ds[PRED_VAR].sel(time=common_time).load(), mask)

    ds_test.close()
    ds_extra.close()
    return common_time, mask, truth.load(), d2m.load(), preds


def make_summary_tables(common_time, mask, truth, d2m, preds):
    rows = []
    daily = pd.DataFrame(index=pd.to_datetime(common_time))
    for name, pred in preds.items():
        rows.append({"Model": name, **overall_metrics(pred, truth, mask)})
        daily[f"{name} RMSE"] = daily_rmse(pred, truth, mask)
        daily[f"{name} MAE"] = daily_mae(pred, truth, mask)
        daily[f"{name} Bias"] = daily_bias(pred, truth, mask)

    summary = pd.DataFrame(rows)
    interp_rmse = float(summary.loc[summary["Model"] == "Interpolation", "RMSE"].iloc[0])
    summary["RMSE improvement vs interpolation (%)"] = (
        (interp_rmse - summary["RMSE"]) / interp_rmse * 100
    )
    daily["Truth Tmax mean (deg C)"] = daily_mean(truth, mask)
    daily["Dew point mean (deg C)"] = daily_mean(d2m, mask)
    daily["Transformer dew-point RMSE benefit"] = (
        daily["Transformer T+Elev RMSE"]
        - daily["Transformer T+Dew Point+Elev RMSE"]
    )
    daily["Transformer T+Elev minus U-Net T+Elev RMSE"] = (
        daily["U-Net T+Elev RMSE"] - daily["Transformer T+Elev RMSE"]
    )
    summary.to_csv(OUT_DIR / "poster_model_summary.csv", index=False)
    daily.to_csv(OUT_DIR / "poster_daily_metrics.csv")
    return summary, daily


def plot_architecture_spatial(mask, truth, preds):
    rmse = {name: spatial_rmse(pred, truth, mask) for name, pred in preds.items()}
    vmax = float(
        np.nanpercentile(
            np.concatenate(
                [
                    rmse["Interpolation"].values.ravel(),
                    rmse["U-Net T+Elev"].values.ravel(),
                    rmse["Transformer T+Elev"].values.ravel(),
                    rmse["Transformer T+Dew Point+Elev"].values.ravel(),
                ]
            ),
            98,
        )
    )
    improvement = rmse["U-Net T+Elev"] - rmse["Transformer T+Elev"]
    lim = float(np.nanpercentile(np.abs(improvement.values), 98))

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), constrained_layout=True)
    add_map(axes[0, 0], rmse["Interpolation"], "Interpolation RMSE", RMSE_CMAP, 0, vmax)
    add_map(axes[0, 1], rmse["U-Net T+Elev"], "U-Net T+Elev RMSE", RMSE_CMAP, 0, vmax)
    add_map(
        axes[1, 0],
        rmse["Transformer T+Elev"],
        "Transformer T+Elev RMSE",
        RMSE_CMAP,
        0,
        vmax,
    )
    add_map(
        axes[1, 1],
        improvement,
        "Transformer advantage over U-Net",
        BENEFIT_CMAP,
        -lim,
        lim,
        "RMSE decrease (deg C)",
    )
    fig.suptitle(
        "Architecture comparison with matched Tmax + elevation inputs",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(OUT_DIR / "architecture_spatial_rmse_poster.png", bbox_inches="tight")
    plt.close(fig)


def plot_architecture_example_field(mask, truth, preds, daily):
    day = daily["Transformer T+Elev minus U-Net T+Elev RMSE"].idxmax()
    time_value = np.datetime64(day)

    truth_day = truth.sel(time=time_value).where(mask)
    unet_day = preds["U-Net T+Elev"].sel(time=time_value).where(mask)
    transformer_day = preds["Transformer T+Elev"].sel(time=time_value).where(mask)
    improvement = (
        np.abs(unet_day - truth_day) - np.abs(transformer_day - truth_day)
    ).where(mask)

    temp_values = np.concatenate(
        [truth_day.values.ravel(), unet_day.values.ravel(), transformer_day.values.ravel()]
    )
    temp_vmin = float(np.nanpercentile(temp_values, 2))
    temp_vmax = float(np.nanpercentile(temp_values, 98))
    improvement_lim = float(np.nanpercentile(np.abs(improvement.values), 98))

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), constrained_layout=False)
    im_temp = None
    for ax, field, title in [
        (axes[0, 0], truth_day, "Truth"),
        (axes[0, 1], unet_day, "U-Net T+Elev prediction"),
        (axes[1, 0], transformer_day, "Transformer T+Elev prediction"),
    ]:
        im_temp = add_map_no_cbar(ax, field, title, TEMP_CMAP, temp_vmin, temp_vmax)

    im_adv = add_map_no_cbar(
        axes[1, 1],
        improvement,
        "Transformer advantage over U-Net",
        BENEFIT_CMAP,
        -improvement_lim,
        improvement_lim,
    )

    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=8, length=2)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)

    fig.subplots_adjust(
        left=0.075,
        right=0.965,
        bottom=0.16,
        top=0.85,
        wspace=0.24,
        hspace=0.42,
    )
    temp_cax = fig.add_axes([0.16, 0.07, 0.42, 0.025])
    adv_cax = fig.add_axes([0.67, 0.07, 0.22, 0.025])
    cbar_temp = fig.colorbar(im_temp, cax=temp_cax, orientation="horizontal")
    cbar_temp.set_label("Daily maximum temperature (deg C)", fontsize=9)
    cbar_temp.ax.tick_params(labelsize=8, length=2)
    cbar_adv = fig.colorbar(im_adv, cax=adv_cax, orientation="horizontal")
    cbar_adv.set_label("Abs. error decrease (deg C); positive = Transformer better", fontsize=9)
    cbar_adv.ax.tick_params(labelsize=8, length=2)

    fig.suptitle(
        f"Architecture comparison on an example day: {day.date()}",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(
        OUT_DIR / "architecture_example_temperature_field_poster.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_average_temperature_and_improvement(mask, truth, preds):
    best_name = "Transformer T+Dew Point+Elev"
    truth_mean = truth.where(mask).mean(dim="time")
    best_mean = preds[best_name].where(mask).mean(dim="time")
    mean_bias = (best_mean - truth_mean).where(mask)
    improvement = (
        spatial_rmse(preds["Interpolation"], truth, mask)
        - spatial_rmse(preds[best_name], truth, mask)
    ).where(mask)

    temp_values = np.concatenate([truth_mean.values.ravel(), best_mean.values.ravel()])
    temp_vmin = float(np.nanpercentile(temp_values, 2))
    temp_vmax = float(np.nanpercentile(temp_values, 98))
    bias_lim = float(np.nanpercentile(np.abs(mean_bias.values), 98))
    improvement_lim = float(np.nanpercentile(np.abs(improvement.values), 98))

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), constrained_layout=False)
    im_temp = None
    for ax, field, title in [
        (axes[0, 0], truth_mean, "Mean observed Tmax"),
        (axes[0, 1], best_mean, "Mean best-model Tmax"),
    ]:
        im_temp = add_map_no_cbar(ax, field, title, TEMP_CMAP, temp_vmin, temp_vmax)

    im_bias = add_map_no_cbar(
        axes[1, 0],
        mean_bias,
        "Mean best-model bias",
        BENEFIT_CMAP + "_r",
        -bias_lim,
        bias_lim,
    )
    im_improve = add_map_no_cbar(
        axes[1, 1],
        improvement,
        "RMSE improvement over interpolation",
        BENEFIT_CMAP,
        -improvement_lim,
        improvement_lim,
    )

    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=8, length=2)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)

    fig.subplots_adjust(
        left=0.075,
        right=0.965,
        bottom=0.17,
        top=0.85,
        wspace=0.24,
        hspace=0.42,
    )
    temp_cax = fig.add_axes([0.16, 0.075, 0.28, 0.025])
    bias_cax = fig.add_axes([0.49, 0.075, 0.20, 0.025])
    improve_cax = fig.add_axes([0.74, 0.075, 0.19, 0.025])
    cbar_temp = fig.colorbar(im_temp, cax=temp_cax, orientation="horizontal")
    cbar_temp.set_label("Mean Tmax (deg C)", fontsize=9)
    cbar_temp.ax.tick_params(labelsize=8, length=2)
    cbar_bias = fig.colorbar(im_bias, cax=bias_cax, orientation="horizontal")
    cbar_bias.set_label("Bias (deg C)", fontsize=9)
    cbar_bias.ax.tick_params(labelsize=8, length=2)
    cbar_improve = fig.colorbar(im_improve, cax=improve_cax, orientation="horizontal")
    cbar_improve.set_label("RMSE decrease (deg C)", fontsize=9)
    cbar_improve.ax.tick_params(labelsize=8, length=2)

    fig.suptitle(
        "Average temperature field and spatial improvement",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(
        OUT_DIR / "average_temperature_field_and_improvement_poster.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_average_architecture_temperature_field(mask, truth, preds):
    truth_mean = truth.where(mask).mean(dim="time")
    unet_mean = preds["U-Net T+Elev"].where(mask).mean(dim="time")
    transformer_mean = preds["Transformer T+Elev"].where(mask).mean(dim="time")
    transformer_advantage = (
        spatial_rmse(preds["U-Net T+Elev"], truth, mask)
        - spatial_rmse(preds["Transformer T+Elev"], truth, mask)
    ).where(mask)

    temp_values = np.concatenate(
        [truth_mean.values.ravel(), unet_mean.values.ravel(), transformer_mean.values.ravel()]
    )
    temp_vmin = float(np.nanpercentile(temp_values, 2))
    temp_vmax = float(np.nanpercentile(temp_values, 98))
    advantage_lim = float(np.nanpercentile(np.abs(transformer_advantage.values), 98))

    fig, axes = plt.subplots(2, 2, figsize=(9.4, 7.0), constrained_layout=False)
    im_temp = None
    for ax, field, title in [
        (axes[0, 0], truth_mean, "Mean observed Tmax"),
        (axes[0, 1], unet_mean, "Mean U-Net T+Elev Tmax"),
        (axes[1, 0], transformer_mean, "Mean Transformer T+Elev Tmax"),
    ]:
        im_temp = add_map_no_cbar(ax, field, title, TEMP_CMAP, temp_vmin, temp_vmax)

    im_adv = add_map_no_cbar(
        axes[1, 1],
        transformer_advantage,
        "Transformer advantage over U-Net",
        BENEFIT_CMAP,
        -advantage_lim,
        advantage_lim,
    )

    for ax in axes.ravel():
        ax.tick_params(axis="both", labelsize=8, length=2)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)

    fig.subplots_adjust(
        left=0.075,
        right=0.965,
        bottom=0.17,
        top=0.85,
        wspace=0.24,
        hspace=0.42,
    )
    temp_cax = fig.add_axes([0.20, 0.075, 0.36, 0.025])
    adv_cax = fig.add_axes([0.68, 0.075, 0.22, 0.025])
    cbar_temp = fig.colorbar(im_temp, cax=temp_cax, orientation="horizontal")
    cbar_temp.set_label("Mean Tmax (deg C)", fontsize=9)
    cbar_temp.ax.tick_params(labelsize=8, length=2)
    cbar_adv = fig.colorbar(im_adv, cax=adv_cax, orientation="horizontal")
    cbar_adv.set_label("RMSE decrease (deg C); positive = Transformer better", fontsize=9)
    cbar_adv.ax.tick_params(labelsize=8, length=2)

    fig.suptitle(
        "Average architecture comparison",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(
        OUT_DIR / "average_architecture_temperature_field_poster.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_dewpoint_spatial(mask, truth, preds):
    rmse_all = {
        name: spatial_rmse(pred, truth, mask)
        for name, pred in preds.items()
    }
    rmse_tf = spatial_rmse(preds["Transformer T+Elev"], truth, mask)
    rmse_d2m = spatial_rmse(preds["Transformer T+Dew Point+Elev"], truth, mask)
    benefit = rmse_tf - rmse_d2m
    vmax = float(
        np.nanpercentile(
            np.concatenate([da.values.ravel() for da in rmse_all.values()]),
            98,
        )
    )
    lim = float(np.nanpercentile(np.abs(benefit.values), 98))

    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.4), constrained_layout=True)
    add_map(axes[0], rmse_tf, "Transformer T+Elev RMSE", RMSE_CMAP, 0, vmax)
    add_map(axes[1], rmse_d2m, "Transformer T+Dew Point+Elev RMSE", RMSE_CMAP, 0, vmax)
    add_map(
        axes[2],
        benefit,
        "Dew-point benefit",
        BENEFIT_CMAP,
        -lim,
        lim,
        "RMSE decrease (deg C)",
    )
    fig.suptitle(
        "Where does dew point improve the Transformer?",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(OUT_DIR / "transformer_dewpoint_benefit_spatial_poster.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 5.9), constrained_layout=False)
    im = add_map_no_cbar(
        ax,
        benefit,
        "Spatial benefit from adding dew point",
        BENEFIT_CMAP,
        -lim,
        lim,
    )
    callouts = [
        (
            -79.25,
            41.6,
            "Allegheny\nPlateau",
            -78.9,
            42.6,
        ),
        (
            -79.1,
            38.7,
            "Central\nAppalachians",
            -77.6,
            39.1,
        ),
        (
            -68.8,
            44.3,
            "Coastal\nMaine",
            -70.5,
            45.4,
        ),
        (
            -73.3,
            42.6,
            "Hudson Valley /\nBerkshires",
            -75.2,
            43.6,
        ),
        (
            -74.0,
            40.7,
            "Urban/coastal\ncorridor",
            -73.7,
            41.4,
        ),
    ]
    for lon, lat, text, tx, ty in callouts:
        ax.scatter(lon, lat, s=42, color="#111111", edgecolor="white", linewidth=0.8, zorder=4)
        ax.annotate(
            text,
            xy=(lon, lat),
            xytext=(tx, ty),
            textcoords="data",
            fontsize=9.2,
            ha="left",
            va="center",
            arrowprops=dict(arrowstyle="->", color="#111111", lw=1.1),
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#333333", alpha=0.88),
        )
    ax.tick_params(axis="both", labelsize=9, length=2)
    ax.set_xlabel("Longitude", fontsize=10)
    ax.set_ylabel("Latitude", fontsize=10)
    fig.subplots_adjust(left=0.10, right=0.80, bottom=0.12, top=0.88)
    cax = fig.add_axes([0.86, 0.20, 0.035, 0.60])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("RMSE decrease (deg C)", fontsize=10)
    cbar.ax.tick_params(labelsize=9, length=2)
    fig.savefig(
        OUT_DIR / "transformer_dewpoint_benefit_annotated_poster.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_dewpoint_weather_bars(daily):
    daily = daily.copy()
    daily["Temperature regime"] = quantile_regime(
        daily["Truth Tmax mean (deg C)"], ["Cool", "Moderate", "Hot"]
    )
    daily["Dew-point regime"] = quantile_regime(
        daily["Dew point mean (deg C)"], ["Dry", "Moderate", "Humid"]
    )

    rows = []
    for col in ["Temperature regime", "Dew-point regime"]:
        for regime, sub in daily.groupby(col, observed=True):
            rows.append(
                {
                    "Condition type": col.replace(" regime", ""),
                    "Regime": str(regime),
                    "n days": int(len(sub)),
                    "Transformer T+Elev RMSE": sub["Transformer T+Elev RMSE"].mean(),
                    "Transformer T+Dew Point+Elev RMSE": sub[
                        "Transformer T+Dew Point+Elev RMSE"
                    ].mean(),
                    "Dew-point RMSE benefit": sub[
                        "Transformer dew-point RMSE benefit"
                    ].mean(),
                }
            )
    condition = pd.DataFrame(rows)
    condition.to_csv(OUT_DIR / "transformer_dewpoint_condition_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.0, 4.5), constrained_layout=True)
    labels = condition["Condition type"] + "\n" + condition["Regime"]
    values = condition["Dew-point RMSE benefit"].values
    colors = ["#2E8B57" if v >= 0 else "#B45F5F" for v in values]
    ax.bar(np.arange(len(values)), values, color=colors, width=0.68)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("RMSE decrease from adding dew point (deg C)")
    ax.set_title("Dew point helps most under specific daily regimes", weight="bold")
    ymin = min(values.min() * 1.35, -0.02)
    ymax = max(values.max() * 1.22, 0.02)
    ax.set_ylim(ymin, ymax)
    for i, (v, n) in enumerate(zip(values, condition["n days"])):
        ax.text(
            i,
            v + (0.006 if v >= 0 else -0.006),
            f"n={n}",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )
    fig.savefig(OUT_DIR / "transformer_dewpoint_weather_benefit_poster.png", bbox_inches="tight")
    plt.close(fig)


def plot_example_day(mask, truth, preds, daily):
    day = daily["Transformer dew-point RMSE benefit"].idxmax()
    time_value = np.datetime64(day)

    fields = [
        ("Truth", truth.sel(time=time_value)),
        ("Interpolation", preds["Interpolation"].sel(time=time_value)),
        ("U-Net T+Elev", preds["U-Net T+Elev"].sel(time=time_value)),
        ("Transformer T+Elev", preds["Transformer T+Elev"].sel(time=time_value)),
        (
            "Transformer T+Dew Point+Elev",
            preds["Transformer T+Dew Point+Elev"].sel(time=time_value),
        ),
    ]
    data_min = float(np.nanpercentile(np.concatenate([f.where(mask).values.ravel() for _, f in fields]), 2))
    data_max = float(np.nanpercentile(np.concatenate([f.where(mask).values.ravel() for _, f in fields]), 98))
    err_lim = float(
        np.nanpercentile(
            np.abs(
                np.concatenate(
                    [
                        (f - truth.sel(time=time_value)).where(mask).values.ravel()
                        for name, f in fields[1:]
                    ]
                )
            ),
            98,
        )
    )

    fig, axes = plt.subplots(2, 5, figsize=(16.2, 6.6), constrained_layout=True)
    for j, (name, field) in enumerate(fields):
        add_map(axes[0, j], field.where(mask), name, TEMP_CMAP, data_min, data_max, "deg C")
        axes[0, j].set_xlabel("")
        if j > 0:
            err = (field - truth.sel(time=time_value)).where(mask)
            add_map(axes[1, j], err, f"{name} error", BENEFIT_CMAP + "_r", -err_lim, err_lim, "deg C")
        else:
            axes[1, j].axis("off")
    fig.suptitle(
        f"Example day where dew point most improves Transformer: {day.date()}",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(OUT_DIR / "example_day_dewpoint_benefit_poster.png", bbox_inches="tight")
    plt.close(fig)


def plot_compact_example_prediction(mask, truth, preds, daily):
    day = daily["Transformer dew-point RMSE benefit"].idxmax()
    time_value = np.datetime64(day)
    best_name = "Transformer T+Dew Point+Elev"

    truth_day = truth.sel(time=time_value).where(mask)
    interp_day = preds["Interpolation"].sel(time=time_value).where(mask)
    best_day = preds[best_name].sel(time=time_value).where(mask)
    best_error = (best_day - truth_day).where(mask)

    temp_fields = [truth_day, interp_day, best_day]
    temp_values = np.concatenate([field.values.ravel() for field in temp_fields])
    temp_vmin = float(np.nanpercentile(temp_values, 2))
    temp_vmax = float(np.nanpercentile(temp_values, 98))
    err_lim = float(np.nanpercentile(np.abs(best_error.values), 98))

    fig, axes = plt.subplots(1, 4, figsize=(12.0, 3.35), constrained_layout=True)
    im_temp = None
    for ax, field, title in zip(
        axes[:3],
        temp_fields,
        ["Truth", "Interpolated ERA5", "Best model prediction"],
    ):
        im_temp = add_map_no_cbar(ax, field, title, TEMP_CMAP, temp_vmin, temp_vmax)

    im_err = add_map_no_cbar(
        axes[3],
        best_error,
        "Best model error",
        BENEFIT_CMAP + "_r",
        -err_lim,
        err_lim,
    )

    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="both", labelsize=8, length=2)
    for ax in axes[1:]:
        ax.set_ylabel("")

    cbar_temp = fig.colorbar(
        im_temp,
        ax=axes[:3],
        orientation="horizontal",
        fraction=0.08,
        pad=0.08,
        aspect=35,
    )
    cbar_temp.set_label("Daily maximum temperature (deg C)", fontsize=10)
    cbar_temp.ax.tick_params(labelsize=8, length=2)
    cbar_err = fig.colorbar(
        im_err,
        ax=axes[3],
        orientation="horizontal",
        fraction=0.08,
        pad=0.08,
        aspect=15,
    )
    cbar_err.set_label("Prediction - truth (deg C)", fontsize=10)
    cbar_err.ax.tick_params(labelsize=8, length=2)

    fig.suptitle(
        f"Example learned temperature field: {day.date()}",
        fontsize=15,
        weight="bold",
    )
    fig.savefig(
        OUT_DIR / "compact_example_temperature_field_poster.png",
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_bad_region_diagnostics(mask, truth, preds):
    topo = xr.open_dataset(TOPO_PATH)
    elevation = topo["elevation"].where(mask)
    slope = topo["slope"].where(mask)
    rmse_best = spatial_rmse(preds["Transformer T+Dew Point+Elev"], truth, mask)
    bias_best = spatial_bias(preds["Transformer T+Dew Point+Elev"], truth, mask)

    records = pd.DataFrame(
        {
            "RMSE": rmse_best.values.ravel(),
            "Bias": bias_best.values.ravel(),
            "Elevation": elevation.values.ravel(),
            "Slope": slope.values.ravel(),
        }
    ).dropna()
    records["Elevation group"] = pd.qcut(records["Elevation"], 4, labels=["Low", "Mid-low", "Mid-high", "High"])
    records["Slope group"] = pd.qcut(records["Slope"], 4, labels=["Low", "Mid-low", "Mid-high", "High"])
    feature_summary = pd.concat(
        [
            records.groupby("Elevation group", observed=True)["RMSE"].mean().rename("Elevation"),
            records.groupby("Slope group", observed=True)["RMSE"].mean().rename("Slope"),
        ],
        axis=1,
    )
    feature_summary.to_csv(OUT_DIR / "bad_region_feature_summary.csv")

    fig = plt.figure(figsize=(12.2, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 1])

    vmax = float(np.nanpercentile(rmse_best.values, 98))
    add_map(
        ax0,
        rmse_best,
        "Best-model spatial RMSE",
        RMSE_CMAP,
        0,
        vmax,
        "RMSE (deg C)",
    )
    feature_summary["Elevation"].plot.bar(ax=ax1, color="#4B83B8")
    ax1.set_title("Error by elevation quartile", weight="bold")
    ax1.set_ylabel("Mean RMSE (deg C)")
    ax1.tick_params(axis="x", rotation=0)
    feature_summary["Slope"].plot.bar(ax=ax2, color="#8E9F3E")
    ax2.set_title("Error by slope quartile", weight="bold")
    ax2.set_ylabel("Mean RMSE (deg C)")
    ax2.tick_params(axis="x", rotation=0)
    for ax, series in [(ax1, feature_summary["Elevation"]), (ax2, feature_summary["Slope"])]:
        ymin = float(series.min() - 0.02)
        ymax = float(series.max() + 0.02)
        ax.set_ylim(ymin, ymax)
        ax.axhline(float(series.mean()), color="#333333", linewidth=0.9, linestyle="--", alpha=0.7)
        for i, value in enumerate(series.values):
            ax.text(
                i,
                value + 0.003,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    fig.suptitle(
        "Where does the best model still struggle?",
        fontsize=16,
        weight="bold",
    )
    fig.savefig(OUT_DIR / "bad_region_diagnostics_poster.png", bbox_inches="tight")
    plt.close(fig)
    topo.close()


def plot_weather_struggle_diagnostics(daily):
    if not WEATHER_DAILY_PATH.exists():
        return

    weather = pd.read_csv(WEATHER_DAILY_PATH, index_col=0, parse_dates=True)
    merged = daily.join(
        weather[
            [
                "Temperature regime",
                "Humidity regime",
                "Rain regime",
                "Wind regime",
                "Cloud regime",
                "Low-cloud regime",
                "Very wet day",
                "Very windy day",
                "Very cloudy day",
                "Hot and humid",
                "Hot and cloudy",
                "Hot and wet",
            ]
        ],
        how="inner",
    )

    rmse_col = "Transformer T+Dew Point+Elev RMSE"
    records = [
        {"Condition": "All days", "n days": len(merged), "Mean RMSE": merged[rmse_col].mean()}
    ]
    condition_masks = {
        "Cloud cover: very high": merged["Very cloudy day"].astype(bool),
        "Cloud cover: high": merged["Cloud regime"].isin(["Cloudy", "Very cloudy"]),
        "Precipitation: high": merged["Rain regime"].isin(["Wet", "Very wet"]),
        "Precipitation: very high": merged["Very wet day"].astype(bool),
        "Wind gust: high": merged["Wind regime"].isin(["Windy", "Very windy"]),
        "Dew point: high": merged["Humidity regime"].eq("Humid"),
        "Temp + dew point: hot/humid": merged["Hot and humid"].astype(bool),
        "Wind gust: very high": merged["Very windy day"].astype(bool),
        "Temperature: hot": merged["Temperature regime"].eq("Hot"),
    }
    for label, mask_values in condition_masks.items():
        sub = merged.loc[mask_values]
        if len(sub) == 0:
            continue
        records.append(
            {
                "Condition": label,
                "n days": len(sub),
                "Mean RMSE": sub[rmse_col].mean(),
            }
        )

    summary = pd.DataFrame(records).sort_values("Mean RMSE", ascending=False)
    summary.to_csv(OUT_DIR / "best_model_weather_struggle_summary.csv", index=False)

    plot_summary = summary.head(8).sort_values("Mean RMSE")
    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    colors = [
        "#B45F5F" if x > summary.loc[summary["Condition"].eq("All days"), "Mean RMSE"].iloc[0] else "#4B83B8"
        for x in plot_summary["Mean RMSE"]
    ]
    ax.barh(plot_summary["Condition"], plot_summary["Mean RMSE"], color=colors)
    all_rmse = summary.loc[summary["Condition"].eq("All days"), "Mean RMSE"].iloc[0]
    ax.axvline(all_rmse, color="#333333", linestyle="--", linewidth=1.1, alpha=0.75)
    ax.set_xlabel("Mean daily RMSE (deg C)")
    ax.set_title("When does the best model still struggle?", weight="bold")
    xmin = max(0, plot_summary["Mean RMSE"].min() - 0.08)
    xmax = plot_summary["Mean RMSE"].max() + 0.08
    ax.set_xlim(xmin, xmax)
    for y, (_, row) in enumerate(plot_summary.iterrows()):
        ax.text(
            row["Mean RMSE"] + 0.008,
            y,
            f"{row['Mean RMSE']:.3f}  n={int(row['n days'])}",
            va="center",
            fontsize=9,
        )
    ax.text(
        all_rmse + 0.006,
        0.02,
        "all-day mean",
        transform=ax.get_xaxis_transform(),
        fontsize=9,
        va="bottom",
        color="#333333",
    )
    ax.text(
        0.01,
        -0.17,
        "Precipitation = daily rain amount; dew point = near-surface moisture/humidity; cloud cover = sky/cloud fraction.",
        transform=ax.transAxes,
        fontsize=9,
        color="#333333",
        ha="left",
    )
    fig.savefig(OUT_DIR / "best_model_weather_struggle_poster.png", bbox_inches="tight")
    plt.close(fig)


def main():
    common_time, mask, truth, d2m, preds = load_data()
    summary, daily = make_summary_tables(common_time, mask, truth, d2m, preds)
    plot_architecture_spatial(mask, truth, preds)
    plot_architecture_example_field(mask, truth, preds, daily)
    plot_average_temperature_and_improvement(mask, truth, preds)
    plot_average_architecture_temperature_field(mask, truth, preds)
    plot_dewpoint_spatial(mask, truth, preds)
    plot_dewpoint_weather_bars(daily)
    plot_example_day(mask, truth, preds, daily)
    plot_compact_example_prediction(mask, truth, preds, daily)
    plot_bad_region_diagnostics(mask, truth, preds)
    plot_weather_struggle_diagnostics(daily)
    print(f"Saved poster figures to {OUT_DIR}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
