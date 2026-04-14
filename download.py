import calendar
from pathlib import Path
from urllib.request import urlretrieve

import cdsapi

# -----------------------------
# Settings
# -----------------------------
YEARS = range(2000, 2026)   # 2000–2025
MONTHS = [6, 7, 8]          # JJA

DOWNLOAD_GRIDMET = True
DOWNLOAD_ERA5 = True

GRIDMET_BASE_URL = "http://www.northwestknowledge.net/metdata/data"
ERA5_DATASET = "derived-era5-single-levels-daily-statistics"


# -----------------------------
# Paths
# -----------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / ".data"
GRIDMET_DIR = DATA_DIR / "GRIDMET"
ERA5_DIR = DATA_DIR / "ERA5"

GRIDMET_DIR.mkdir(parents=True, exist_ok=True)
ERA5_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# GRIDMET
# -----------------------------
def download_gridmet_tmmx_for_year(year: int, output_dir: Path) -> None:
    target_file = output_dir / f"tmmx_{year}.nc"

    if target_file.exists():
        print(f"[GRIDMET] Skipping existing {target_file.name}")
        return

    url = f"{GRIDMET_BASE_URL}/tmmx_{year}.nc"
    print(f"[GRIDMET] Downloading {url} ...")
    urlretrieve(url, target_file)
    print(f"[GRIDMET] Saved {target_file}")


# -----------------------------
# ERA5 helpers
# -----------------------------
def days_for_month(year: int, month: int) -> list[str]:
    last_day = calendar.monthrange(year, month)[1]
    return [f"{day:02d}" for day in range(1, last_day + 1)]


def month_name(month: int) -> str:
    return calendar.month_abbr[month].lower()   # jun, jul, aug


def era5_output_filename(year: int, month: int) -> str:
    return (
        f"era5_t2m_dailymax_"
        f"{year}_{month:02d}_{month_name(month)}_"
        f"utc-04_47N-37N_80W-60W.nc"
    )


def build_era5_request(year: int, month: int) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": ["2m_temperature"],
        "year": str(year),
        "month": [f"{month:02d}"],
        "day": days_for_month(year, month),
        "daily_statistic": "daily_maximum",
        "time_zone": "utc-04:00",
        "frequency": "1_hourly",
        "area": [47, -80, 37, -60],   # North, West, South, East
    }


def download_era5_for_month(client: cdsapi.Client, year: int, month: int, output_dir: Path) -> None:
    target_file = output_dir / era5_output_filename(year, month)

    if target_file.exists():
        print(f"[ERA5] Skipping existing {target_file.name}")
        return

    request = build_era5_request(year, month)
    print(f"[ERA5] Downloading {year}-{month:02d} ...")

    # synchronous request: waits until finished, then saves to target_file
    client.retrieve(ERA5_DATASET, request, str(target_file))

    print(f"[ERA5] Saved {target_file}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    if DOWNLOAD_GRIDMET:
        print("\n=== Downloading GRIDMET ===")
        for year in YEARS:
            try:
                download_gridmet_tmmx_for_year(year, GRIDMET_DIR)
            except Exception as e:
                print(f"[GRIDMET] Failed for {year}: {e}")

    if DOWNLOAD_ERA5:
        print("\n=== Downloading ERA5 ===")
        client = cdsapi.Client()

        for year in YEARS:
            for month in MONTHS:
                try:
                    download_era5_for_month(client, year, month, ERA5_DIR)
                except Exception as e:
                    print(f"[ERA5] Failed for {year}-{month:02d}: {e}")


if __name__ == "__main__":
    main()