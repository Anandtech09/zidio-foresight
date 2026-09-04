"""Data ingestion, cleaning, and reconciliation pipeline.

Consolidates sales transactions, catalog metadata, calendar schedules, and inventory positions
into a normalized analysis-ready dataset with full audit trail logging.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from src.data_fetcher import fetch_datasets
from src.utils import (
    DATA_CLEANED,
    DATA_FEATURES,
    DATA_RAW,
    cap_outliers,
    detect_outliers_iqr,
    get_logger,
)

logger = get_logger("pipeline")

SALES_COLUMNS = ["date", "sku_id", "units_sold", "revenue", "unit_price", "promo_flag"]
SKU_COLUMNS = ["sku_id", "category", "subcategory", "launch_date", "unit_cost", "list_price"]
CALENDAR_COLUMNS = ["date", "week", "month", "season", "is_holiday", "promo_event"]
INVENTORY_COLUMNS = ["date", "sku_id", "on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]

cleaning_log: List[Dict[str, object]] = []


def record_decision(step: str, detail: str, rows_affected: int = 0) -> None:
    """Record an audit trail event for data transformations."""
    cleaning_log.append({"step": step, "detail": detail, "rows_affected": rows_affected})
    logger.info("[%s] %s (affected: %d)", step, detail, rows_affected)


def ingest_raw() -> Dict[str, pd.DataFrame]:
    """Load raw source CSV tables from storage, fetching missing files if required."""
    fetch_datasets()

    file_mapping = {
        "sales_daily": DATA_RAW / "sales_daily.csv",
        "sku_master": DATA_RAW / "sku_master.csv",
        "calendar": DATA_RAW / "calendar.csv",
        "inventory_snapshots": DATA_RAW / "inventory_snapshots.csv",
    }

    tables: Dict[str, pd.DataFrame] = {}
    for name, path in file_mapping.items():
        if not path.exists():
            logger.error("Required raw file not found: %s", path)
            raise FileNotFoundError(f"Missing raw table: {path}")
        df = pd.read_csv(path)
        logger.info("Loaded %s: %d rows, %d cols", name, len(df), len(df.columns))
        tables[name] = df

    return tables


def preprocess_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate daily sales transactions."""
    initial_rows = len(df)
    column_mapping = {
        "Date": "date", "date": "date",
        "SKU_ID": "sku_id", "Product_ID": "sku_id", "sku_id": "sku_id",
        "product_id": "sku_id", "item_id": "sku_id",
        "Units_Sold": "units_sold", "units_sold": "units_sold",
        "quantity": "units_sold", "sales": "units_sold",
        "Revenue": "revenue", "revenue": "revenue",
        "Unit_Price": "unit_price", "unit_price": "unit_price",
        "price": "unit_price", "selling_price": "unit_price",
        "Promo_Flag": "promo_flag", "promo_flag": "promo_flag",
        "promotion": "promo_flag", "is_promotion": "promo_flag",
    }
    data = df.rename(columns={c: column_mapping.get(c, c) for c in df.columns})

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    invalid_dates = int(data["date"].isna().sum())
    if invalid_dates > 0:
        record_decision("Invalid Dates", f"Dropped {invalid_dates} unparseable timestamps", invalid_dates)
        data = data.dropna(subset=["date"])

    for col in ["units_sold", "revenue", "unit_price"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    if "promo_flag" in data.columns:
        data["promo_flag"] = data["promo_flag"].fillna(0).astype(int)
    else:
        data["promo_flag"] = 0
        record_decision("Default Promo", "Defaulted absent promo_flag to 0", len(data))

    duplicate_rows = int(data.duplicated(subset=["date", "sku_id"], keep="first").sum())
    if duplicate_rows > 0:
        data = data.drop_duplicates(subset=["date", "sku_id"], keep="first")
        record_decision("Deduplicate Sales", f"Removed {duplicate_rows} duplicate date-SKU records", duplicate_rows)

    if "units_sold" in data.columns:
        missing_units = int(data["units_sold"].isna().sum())
        if missing_units > 0:
            data["units_sold"] = data.groupby("sku_id")["units_sold"].transform(lambda s: s.fillna(s.median()))
            data["units_sold"] = data["units_sold"].fillna(0)
            record_decision("Impute Units Sold", f"Imputed {missing_units} values using SKU median", missing_units)

    if "revenue" in data.columns and "unit_price" in data.columns:
        missing_rev = int(data["revenue"].isna().sum())
        if missing_rev > 0:
            data["revenue"] = data["revenue"].fillna(data["units_sold"] * data["unit_price"])
            record_decision("Impute Revenue", f"Derived {missing_rev} values via price multiplication", missing_rev)

    if "unit_price" in data.columns:
        missing_price = int(data["unit_price"].isna().sum())
        if missing_price > 0:
            data["unit_price"] = data.groupby("sku_id")["unit_price"].transform(lambda s: s.ffill().bfill())
            record_decision("Impute Price", f"Forward/backfilled {missing_price} prices per SKU", missing_price)

    if "units_sold" in data.columns:
        outlier_mask = detect_outliers_iqr(data["units_sold"])
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            data["units_sold"] = cap_outliers(data["units_sold"])
            record_decision("Cap Outliers", f"Capped {outlier_count} volume outliers via 1.5x IQR", outlier_count)

    for numeric_col in ["units_sold", "revenue"]:
        if numeric_col in data.columns:
            negative_count = int((data[numeric_col] < 0).sum())
            if negative_count > 0:
                data[numeric_col] = data[numeric_col].clip(lower=0)
                record_decision("Clip Negatives", f"Clipped {negative_count} negative entries in {numeric_col}", negative_count)

    data = data.sort_values(["sku_id", "date"]).reset_index(drop=True)
    record_decision("Sales Ingestion Summary", f"Reconciled {initial_rows} -> {len(data)} rows across {data['sku_id'].nunique()} SKUs")
    return data


def preprocess_sku_master(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize SKU catalog taxonomy, dates, and baseline pricing."""
    column_mapping = {
        "SKU_ID": "sku_id", "Product_ID": "sku_id",
        "Category": "category", "category": "category",
        "Subcategory": "subcategory", "subcategory": "subcategory",
        "sub_category": "subcategory",
        "Launch_Date": "launch_date", "launch_date": "launch_date",
        "Unit_Cost": "unit_cost", "unit_cost": "unit_cost", "cost": "unit_cost",
        "List_Price": "list_price", "list_price": "list_price", "price": "list_price",
    }
    data = df.rename(columns={c: column_mapping.get(c, c) for c in df.columns})

    duplicates = int(data.duplicated(subset=["sku_id"], keep="first").sum())
    if duplicates > 0:
        data = data.drop_duplicates(subset=["sku_id"], keep="first")
        record_decision("Deduplicate SKU Master", f"Removed {duplicates} duplicate SKU rows", duplicates)

    for label_col in ["category", "subcategory"]:
        if label_col in data.columns:
            data[label_col] = data[label_col].astype(str).str.strip().str.title()

    if "launch_date" in data.columns:
        data["launch_date"] = pd.to_datetime(data["launch_date"], errors="coerce")

    for price_col in ["unit_cost", "list_price"]:
        if price_col in data.columns:
            data[price_col] = pd.to_numeric(data[price_col], errors="coerce")

    return data


def preprocess_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize date attributes, promotional events, and holiday indicators."""
    column_mapping = {
        "Date": "date",
        "Week": "week", "week_number": "week",
        "Month": "month",
        "Season": "season",
        "Is_Holiday": "is_holiday", "holiday": "is_holiday",
        "Promo_Event": "promo_event", "event": "promo_event",
    }
    data = df.rename(columns={c: column_mapping.get(c, c) for c in df.columns})

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])

    if "week" not in data.columns:
        data["week"] = data["date"].dt.isocalendar().week.astype(int)
    if "month" not in data.columns:
        data["month"] = data["date"].dt.month

    season_mapping = {
        1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
        5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter"
    }
    if "season" not in data.columns:
        data["season"] = data["month"].map(season_mapping)
        record_decision("Derive Season", "Mapped quarterly seasons from calendar month", len(data))

    if "is_holiday" in data.columns:
        data["is_holiday"] = data["is_holiday"].fillna(0).astype(int)
    else:
        data["is_holiday"] = 0

    if "promo_event" in data.columns:
        data["promo_event"] = data["promo_event"].fillna("").astype(str)
    else:
        data["promo_event"] = ""

    return data.sort_values("date").reset_index(drop=True)


def preprocess_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile snapshot inventory positions, supplier lead times, and replenishment points."""
    column_mapping = {
        "Date": "date", "SKU_ID": "sku_id", "Product_ID": "sku_id",
        "On_Hand": "on_hand_units", "on_hand": "on_hand_units",
        "stock": "on_hand_units", "inventory_level": "on_hand_units",
        "On_Order": "on_order_units", "on_order": "on_order_units",
        "Lead_Time": "lead_time_days", "lead_time": "lead_time_days",
        "Reorder_Point": "reorder_point", "reorder_level": "reorder_point",
    }
    data = df.rename(columns={c: column_mapping.get(c, c) for c in df.columns})

    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data = data.dropna(subset=["date"])

    if "lead_time_days" in data.columns:
        missing_lt = int(data["lead_time_days"].isna().sum())
        if missing_lt > 0:
            data["lead_time_days"] = data.groupby("sku_id")["lead_time_days"].transform(lambda s: s.ffill().bfill())
            data["lead_time_days"] = data["lead_time_days"].fillna(data["lead_time_days"].median())
            record_decision("Impute Lead Time", f"Imputed {missing_lt} missing lead time values", missing_lt)

    if "on_order_units" in data.columns:
        data["on_order_units"] = data["on_order_units"].fillna(0)
    else:
        data["on_order_units"] = 0

    if "reorder_point" in data.columns:
        data["reorder_point"] = data["reorder_point"].fillna(20)
    else:
        data["reorder_point"] = 20

    for col in ["on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]:
        if col in data.columns:
            data[col] = data[col].clip(lower=0)

    return data.sort_values(["sku_id", "date"]).reset_index(drop=True)


def build_analysis_dataset(
    sales: pd.DataFrame,
    sku_master: pd.DataFrame,
    calendar: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Construct unified analysis dataset via relational and backward temporal as-of joins."""
    sales["date"] = pd.to_datetime(sales["date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    inventory["date"] = pd.to_datetime(inventory["date"])

    merged = sales.merge(sku_master, on="sku_id", how="left")
    merged = merged.merge(calendar, on="date", how="left")

    inv_cols = ["date", "sku_id", "on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]
    merged = pd.merge_asof(
        merged.sort_values("date"),
        inventory.sort_values("date")[inv_cols],
        on="date",
        by="sku_id",
        direction="backward",
    )

    for col in ["on_hand_units", "on_order_units", "lead_time_days", "reorder_point"]:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0)

    return merged.sort_values(["sku_id", "date"]).reset_index(drop=True)


def save_cleaning_log() -> None:
    """Serialize the transformation audit trail to storage."""
    if cleaning_log:
        log_df = pd.DataFrame(cleaning_log)
        log_path = DATA_CLEANED / "cleaning_log.csv"
        log_df.to_csv(log_path, index=False)
        logger.info("Saved %d transformation decisions to %s", len(cleaning_log), log_path)


def run_pipeline() -> pd.DataFrame:
    """Execute complete ingestion, validation, and feature preparation pipeline."""
    tables = ingest_raw()

    sales = preprocess_sales(tables["sales_daily"])
    sku_master = preprocess_sku_master(tables["sku_master"])
    calendar = preprocess_calendar(tables["calendar"])
    inventory = preprocess_inventory(tables["inventory_snapshots"])

    sales.to_csv(DATA_CLEANED / "sales_daily_clean.csv", index=False)
    sku_master.to_csv(DATA_CLEANED / "sku_master_clean.csv", index=False)
    calendar.to_csv(DATA_CLEANED / "calendar_clean.csv", index=False)
    inventory.to_csv(DATA_CLEANED / "inventory_clean.csv", index=False)

    analysis_df = build_analysis_dataset(sales, sku_master, calendar, inventory)
    analysis_df.to_csv(DATA_FEATURES / "analysis_ready.csv", index=False)

    save_cleaning_log()
    logger.info("Pipeline complete. Output saved to %s (%d rows)", DATA_FEATURES / "analysis_ready.csv", len(analysis_df))
    return analysis_df


if __name__ == "__main__":
    run_pipeline()
