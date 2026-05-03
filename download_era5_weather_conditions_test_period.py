# download_era5_weather_conditions_test_period.py

import calendar
from pathlib import Path
import cdsapi

DATASET = "derived-era5-single-levels-daily-statistics"

PROJECT_DIR = Path(".").resolve()
OUT_DIR = PROJECT_DIR / ".data" / "ERA5_weather_conditions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

YEARS = range(2022, 2026)
MONTHS = [6, 7, 8]

AREA = [47, -80, 37, -60]  # North, West, South, East

def days_for_month(year, month):
    last_day = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, last_day + 1)]

def month_name(month):
    return calendar.month_abbr[month].lower()

def download_month(client, year, month, variables, daily_statistic, label):
    out_file = OUT_DIR / (
        f"era5_{label}_{year}_{month:02d}_{month_name(month)}_"
        f"utc-04_47N-37N_80W-60W.nc"
    )

    if out_file.exists():
        print(f"Skipping existing {out_file}")
        return

    request = {
        "product_type": "reanalysis",
        "variable": variables,
        "year": str(year),
        "month": [f"{month:02d}"],
        "day": days_for_month(year, month),
        "daily_statistic": daily_statistic,
        "time_zone": "utc-04:00",   # match your existing ERA5 preprocessing
        "frequency": "1_hourly",
        "area": AREA,
    }

    print(f"Downloading {label}: {year}-{month:02d}")
    client.retrieve(DATASET, request, str(out_file))
    print(f"Saved {out_file}")

client = cdsapi.Client()

for year in YEARS:
    for month in MONTHS:
        # Cloudiness: daily mean is appropriate.
        download_month(
            client,
            year,
            month,
            variables=[
                "low_cloud_cover",
                "total_cloud_cover",
            ],
            daily_statistic="daily_mean",
            label="cloud_dailymean",
        )

        # Gust: daily maximum is more meaningful than daily mean.
        download_month(
            client,
            year,
            month,
            variables=[
                "10m_wind_gust_since_previous_post_processing",
            ],
            daily_statistic="daily_maximum",
            label="gust_dailymax",
        )

        # Try daily_sum for precipitation. If CDS rejects this, use daily_mean
        # or download hourly ERA5 and aggregate manually.
        download_month(
            client,
            year,
            month,
            variables=[
                "total_precipitation",
            ],
            daily_statistic="daily_sum",
            label="precip_dailysum",
        )
