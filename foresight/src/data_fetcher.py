"""
data_fetcher.py — Download real-world retail datasets from Kaggle.

Fetches multiple datasets and saves them as raw CSVs for the pipeline.
Uses kagglehub when available, falls back to direct URL download.
"""

import os
import zipfile
import requests
import pandas as pd
from pathlib import Path
from src.utils import get_logger, DATA_RAW

logger = get_logger("data_fetcher")

# ---------------------------------------------------------------------------
# Dataset registry — sources we pull from
# ---------------------------------------------------------------------------
DATASETS = {
    "retail_inventory": {
        "kaggle_id": "karrrimbaa/retail-store-inventory-forecasting-dataset",
        "description": "Retail Store Inventory Forecasting — 73K+ rows, daily sales/inventory",
        "fallback_url": None,  # Kaggle API required
    },
}

# Direct download URLs for datasets that don't require Kaggle auth
DIRECT_URLS = {
    "sample_sales": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/shampoo.csv",
}


def download_via_kagglehub(dataset_id: str, output_dir: Path) -> Path:
    """Download a dataset using the kagglehub library."""
    try:
        import kagglehub
        logger.info(f"Downloading via kagglehub: {dataset_id}")
        path = kagglehub.dataset_download(dataset_id)
        logger.info(f"Downloaded to: {path}")
        return Path(path)
    except ImportError:
        logger.warning("kagglehub not installed. Trying kaggle CLI...")
        return download_via_kaggle_cli(dataset_id, output_dir)
    except Exception as e:
        logger.error(f"kagglehub download failed: {e}")
        return download_via_kaggle_cli(dataset_id, output_dir)


def download_via_kaggle_cli(dataset_id: str, output_dir: Path) -> Path:
    """Download a dataset using the kaggle CLI tool."""
    try:
        import subprocess
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = f"kaggle datasets download -d {dataset_id} -p {output_dir} --unzip"
        logger.info(f"Running: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"CLI download successful to {output_dir}")
            return output_dir
        else:
            logger.error(f"CLI failed: {result.stderr}")
            raise RuntimeError(result.stderr)
    except Exception as e:
        logger.error(f"Kaggle CLI download failed: {e}")
        raise


def download_direct(url: str, output_path: Path) -> Path:
    """Download a file directly from a URL."""
    logger.info(f"Downloading from {url}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    logger.info(f"Saved to {output_path} ({output_path.stat().st_size:,} bytes)")
    return output_path


def extract_zip(zip_path: Path, output_dir: Path) -> None:
    """Extract a zip file to the specified directory."""
    logger.info(f"Extracting {zip_path} to {output_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output_dir)
    logger.info(f"Extracted {len(list(output_dir.glob('*')))} files")


def generate_synthetic_data() -> dict:
    """
    Generate synthetic data matching the brief's schema as a fallback
    when Kaggle datasets cannot be downloaded.

    This creates realistic data with:
    - ~200 SKUs across 5 categories
    - 2 years of daily sales history
    - Seasonality, trends, promotions
    - Deliberate data quality issues (missing values, duplicates)
    """
    import numpy as np

    logger.info("Generating synthetic data as fallback...")
    np.random.seed(42)

    # --- Date range: 2 years ---
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    n_days = len(dates)

    # --- SKU Master ---
    categories = {
        "Furnishings": ["Sofas", "Tables", "Chairs", "Beds"],
        "Decor": ["Vases", "Wall Art", "Candles", "Mirrors"],
        "Kitchen": ["Cookware", "Utensils", "Storage Jars", "Appliances"],
        "Lighting": ["Lamps", "Ceiling Lights", "Fairy Lights", "Candle Holders"],
        "Storage": ["Shelves", "Baskets", "Boxes", "Hooks"],
    }

    sku_records = []
    sku_id = 1
    for cat, subcats in categories.items():
        for subcat in subcats:
            n_skus = np.random.randint(8, 12)
            for _ in range(n_skus):
                cost = round(np.random.uniform(100, 5000), 2)
                markup = np.random.uniform(1.3, 2.5)
                sku_records.append({
                    "sku_id": f"SKU_{sku_id:04d}",
                    "category": cat,
                    "subcategory": subcat,
                    "launch_date": str(dates[0] + pd.Timedelta(days=np.random.randint(0, 180))),
                    "unit_cost": cost,
                    "list_price": round(cost * markup, 2),
                })
                sku_id += 1

    sku_master = pd.DataFrame(sku_records)
    n_skus = len(sku_master)
    logger.info(f"Generated {n_skus} SKUs across {len(categories)} categories")

    # --- Sales Daily ---
    sales_records = []
    for _, sku in sku_master.iterrows():
        base_demand = np.random.uniform(1, 30)
        launch = pd.Timestamp(sku["launch_date"])

        for date in dates:
            if date < launch:
                continue

            # Weekly seasonality (weekend boost)
            dow_effect = 1.3 if date.dayofweek >= 5 else 1.0

            # Monthly seasonality
            month_effect = 1.0 + 0.3 * np.sin(2 * np.pi * date.month / 12)

            # Festive season boost (Oct-Dec)
            festive = 1.4 if date.month in [10, 11, 12] else 1.0

            # Promo flag (15% of days)
            is_promo = 1 if np.random.random() < 0.15 else 0
            promo_effect = 1.25 if is_promo else 1.0

            # Demand with noise
            demand = base_demand * dow_effect * month_effect * festive * promo_effect
            demand = max(0, int(demand + np.random.normal(0, demand * 0.2)))

            price = sku["list_price"] * (0.85 if is_promo else 1.0)
            revenue = round(demand * price, 2)

            sales_records.append({
                "date": str(date.date()),
                "sku_id": sku["sku_id"],
                "units_sold": demand,
                "revenue": revenue,
                "unit_price": round(price, 2),
                "promo_flag": is_promo,
            })

    sales_daily = pd.DataFrame(sales_records)

    # Inject data quality issues
    # Missing values (~2% of units_sold)
    mask = np.random.random(len(sales_daily)) < 0.02
    sales_daily.loc[mask, "units_sold"] = np.nan

    # Duplicate rows (~0.5%)
    n_dupes = int(len(sales_daily) * 0.005)
    dupes = sales_daily.sample(n=n_dupes, random_state=42)
    sales_daily = pd.concat([sales_daily, dupes], ignore_index=True)

    logger.info(f"Generated {len(sales_daily)} sales records (incl. {n_dupes} deliberate dupes)")

    # --- Calendar ---
    indian_holidays = {
        "2023-01-26": "Republic Day",
        "2023-03-08": "Holi",
        "2023-08-15": "Independence Day",
        "2023-10-24": "Dussehra",
        "2023-11-12": "Diwali",
        "2023-12-25": "Christmas",
        "2024-01-26": "Republic Day",
        "2024-03-25": "Holi",
        "2024-08-15": "Independence Day",
        "2024-10-12": "Dussehra",
        "2024-11-01": "Diwali",
        "2024-12-25": "Christmas",
    }

    promo_events = {
        "2023-01-20": "Republic Day Sale",
        "2023-07-15": "Prime Day Sale",
        "2023-10-08": "Big Billion Days",
        "2023-11-24": "Black Friday",
        "2024-01-20": "Republic Day Sale",
        "2024-07-20": "Prime Day Sale",
        "2024-10-06": "Big Billion Days",
        "2024-11-29": "Black Friday",
    }

    season_map = {1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
                  5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
                  9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter"}

    calendar_records = []
    for date in dates:
        ds = str(date.date())
        calendar_records.append({
            "date": ds,
            "week": date.isocalendar()[1],
            "month": date.month,
            "season": season_map[date.month],
            "is_holiday": 1 if ds in indian_holidays else 0,
            "promo_event": promo_events.get(ds, ""),
        })
    calendar_df = pd.DataFrame(calendar_records)

    # --- Inventory Snapshots (weekly) ---
    snapshot_dates = pd.date_range("2023-01-01", "2024-12-31", freq="W")
    inv_records = []
    for _, sku in sku_master.iterrows():
        on_hand = np.random.randint(50, 500)
        lead_time = np.random.choice([7, 14, 21, 28])
        reorder_point = np.random.randint(20, 100)

        for snap_date in snapshot_dates:
            sold_this_week = np.random.randint(0, int(on_hand * 0.3) + 1)
            on_hand = max(0, on_hand - sold_this_week)

            # Reorder if below reorder point
            on_order = 0
            if on_hand < reorder_point:
                on_order = np.random.randint(50, 200)
                # Stock arrives after lead time (simplified)
                if np.random.random() > 0.5:
                    on_hand += on_order
                    on_order = 0

            inv_records.append({
                "date": str(snap_date.date()),
                "sku_id": sku["sku_id"],
                "on_hand_units": on_hand,
                "on_order_units": on_order,
                "lead_time_days": lead_time,
                "reorder_point": reorder_point,
            })

    inventory_snapshots = pd.DataFrame(inv_records)

    # Some missing lead_time_days (~3%)
    mask = np.random.random(len(inventory_snapshots)) < 0.03
    inventory_snapshots.loc[mask, "lead_time_days"] = np.nan

    logger.info(f"Generated {len(inventory_snapshots)} inventory snapshots")

    return {
        "sales_daily": sales_daily,
        "sku_master": sku_master,
        "calendar": calendar_df,
        "inventory_snapshots": inventory_snapshots,
    }


def fetch_datasets() -> dict:
    """
    Main entry point: download datasets from Kaggle or generate synthetic fallback.

    Returns dict of {table_name: Path_to_csv}.
    """
    output_files = {
        "sales_daily": DATA_RAW / "sales_daily.csv",
        "sku_master": DATA_RAW / "sku_master.csv",
        "calendar": DATA_RAW / "calendar.csv",
        "inventory_snapshots": DATA_RAW / "inventory_snapshots.csv",
    }

    # Check if data already exists (idempotent)
    all_exist = all(f.exists() for f in output_files.values())
    if all_exist:
        logger.info("All raw data files already exist. Skipping download.")
        return output_files

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    # Try Kaggle download first
    try:
        kaggle_id = DATASETS["retail_inventory"]["kaggle_id"]
        downloaded_path = download_via_kagglehub(kaggle_id, DATA_RAW)

        # Find and load the downloaded CSV(s)
        csv_files = list(Path(downloaded_path).glob("*.csv"))
        if csv_files:
            logger.info(f"Found {len(csv_files)} CSV files from Kaggle download")
            # The Kaggle dataset has a single large CSV — we'll split it in pipeline.py
            for csv_file in csv_files:
                # Copy to our raw directory
                import shutil
                dest = DATA_RAW / csv_file.name
                if not dest.exists():
                    shutil.copy2(csv_file, dest)
                    logger.info(f"Copied {csv_file.name} to {dest}")

            logger.info("Kaggle download successful!")
            return output_files

    except Exception as e:
        logger.warning(f"Kaggle download failed: {e}")
        logger.info("Falling back to synthetic data generation...")

    # Fallback: generate synthetic data
    data = generate_synthetic_data()
    for name, df in data.items():
        path = output_files[name]
        df.to_csv(path, index=False)
        logger.info(f"Saved {name}: {len(df)} rows -> {path}")

    logger.info("Data fetching complete!")
    return output_files


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fetch_datasets()
