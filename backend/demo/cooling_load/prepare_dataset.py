"""Build the cooling-load raw and platform CSVs from EnergyPlus outputs.

Sources (read-only):
  * EPW weather: CHN_AH_Anqing.584240_TMYx.2011-2025.epw
  * EnergyPlus hourly output: eplusout.csv from the Anqing simulation

Outputs mirror the electric-load pipeline layout:
  * cooling_load_raw_with_lags.csv  (scenario + features + lag_1/lag_24 + target)
  * cooling_load_platform.csv       (first column timestamp, last column target)

The cooling target is the sum of every ``Air System Total Cooling Energy``
hourly column, converted from J/h to average W by dividing by 3600 s.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_EPW = Path("weather/CHN_AH_Anqing.584240_TMYx.2011-2025.epw")
DEFAULT_SIM_CSV = Path("sim/eplusout.csv")
EPW = DEFAULT_EPW
SIM_CSV = DEFAULT_SIM_CSV
RAW_OUT = Path("cooling_load_raw_with_lags.csv")
PLATFORM_OUT = Path("cooling_load_platform.csv")
SUMMARY_OUT = Path("cooling_dataset_summary.json")

TARGET = "target_total_cooling_energy_w"
SCENARIO = "ANQ"  # Anqing, single scenario

WEATHER_COLUMNS = [
    "year", "month", "day", "hour", "minute", "source_flags",
    "dry_bulb_temp_c", "dew_point_temp_c", "relative_humidity_pct",
    "pressure_pa", "extraterrestrial_horizontal_radiation_wh_m2",
    "extraterrestrial_direct_normal_radiation_wh_m2",
    "horizontal_infrared_radiation_wh_m2", "global_horizontal_radiation_wh_m2",
    "direct_normal_radiation_wh_m2", "diffuse_horizontal_radiation_wh_m2",
    "global_horizontal_illuminance_lux", "direct_normal_illuminance_lux",
    "diffuse_horizontal_illuminance_lux", "zenith_luminance_cd_m2",
    "wind_direction_deg", "wind_speed_m_s", "total_sky_cover_tenths",
    "opaque_sky_cover_tenths", "visibility_km", "ceiling_height_m",
    "present_weather_observation", "present_weather_codes",
    "precipitable_water_mm", "aerosol_optical_depth", "snow_depth_cm",
    "days_since_last_snowfall", "albedo", "liquid_precipitation_depth_mm",
    "liquid_precipitation_quantity_hr",
]

FEATURES = [
    "month", "day", "hour", "day_of_week", "day_of_year", "is_weekend",
    "dry_bulb_temp_c", "dew_point_temp_c", "relative_humidity_pct", "pressure_pa",
    "global_horizontal_radiation_wh_m2", "direct_normal_radiation_wh_m2",
    "diffuse_horizontal_radiation_wh_m2", "wind_direction_deg", "wind_speed_m_s",
    "total_sky_cover_tenths", "opaque_sky_cover_tenths", "snow_depth_cm",
    "lag_1", "lag_24",
]

VALID_RANGES = {
    "dry_bulb_temp_c": (-90, 70),
    "dew_point_temp_c": (-90, 70),
    "relative_humidity_pct": (0, 100),
    "pressure_pa": (80000, 110000),
    "global_horizontal_radiation_wh_m2": (0, 2000),
    "direct_normal_radiation_wh_m2": (0, 2000),
    "diffuse_horizontal_radiation_wh_m2": (0, 2000),
    "wind_direction_deg": (0, 360),
    "wind_speed_m_s": (0, 100),
    "total_sky_cover_tenths": (0, 10),
    "opaque_sky_cover_tenths": (0, 10),
    "snow_depth_cm": (0, 1000),
}

REFERENCE_YEAR = 2001
PLATFORM_EPOCH = pd.Timestamp("2000-01-01 00:00:00")


def parse_epw(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        path, skiprows=8, header=None, names=WEATHER_COLUMNS,
        encoding="utf-8-sig", dtype=str, keep_default_na=False,
    )
    if len(frame) != 8760:
        raise ValueError(f"{path.name}: expected 8760 weather rows, got {len(frame)}")
    selected = [
        "month", "day", "hour", "dry_bulb_temp_c", "dew_point_temp_c",
        "relative_humidity_pct", "pressure_pa", "global_horizontal_radiation_wh_m2",
        "direct_normal_radiation_wh_m2", "diffuse_horizontal_radiation_wh_m2",
        "wind_direction_deg", "wind_speed_m_s", "total_sky_cover_tenths",
        "opaque_sky_cover_tenths", "snow_depth_cm", "liquid_precipitation_depth_mm",
    ]
    weather = frame[selected].apply(pd.to_numeric, errors="coerce")
    for column, (lower, upper) in VALID_RANGES.items():
        weather.loc[
            (weather[column] < lower) | (weather[column] > upper), column
        ] = np.nan
    return weather


def parse_cooling(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    cooling_columns = [c for c in frame.columns if "Total Cooling Energy" in c]
    if not cooling_columns:
        raise KeyError(f"{path.name}: no cooling energy columns found")
    if len(frame) != 8760:
        raise ValueError(f"{path.name}: expected 8760 energy rows, got {len(frame)}")

    raw_time = frame["Date/Time"].astype(str).str.strip()
    parts = raw_time.str.extract(r"^(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})$")
    if parts.isna().any(axis=None):
        raise ValueError(f"{path.name}: unable to parse Date/Time")
    month, day, hour = parts[0].astype(int), parts[1].astype(int), parts[2].astype(int)

    joules = frame[cooling_columns].apply(pd.to_numeric, errors="coerce")
    if joules.isna().any(axis=None):
        raise ValueError(f"{path.name}: cooling energy contains non-numeric values")
    if (joules < 0).any(axis=None):
        raise ValueError(f"{path.name}: cooling energy contains negative values")
    total_watts = joules.sum(axis=1) / 3600.0
    datacenter_columns = [c for c in cooling_columns if "DATACENTER" in c.upper()]
    datacenter_watts = joules[datacenter_columns].sum(axis=1) / 3600.0
    building_watts = total_watts - datacenter_watts

    date = pd.to_datetime({"year": REFERENCE_YEAR, "month": month, "day": day})
    return pd.DataFrame({
        "month": month.to_numpy(),
        "day": day.to_numpy(),
        "hour": hour.to_numpy(),
        "day_of_week": date.dt.dayofweek.to_numpy(),
        "day_of_year": date.dt.dayofyear.to_numpy(),
        "is_weekend": (date.dt.dayofweek >= 5).astype(int).to_numpy(),
        TARGET: total_watts.to_numpy(),
        "building_cooling_energy_w": building_watts.to_numpy(),
        "datacenter_cooling_energy_w": datacenter_watts.to_numpy(),
    }), cooling_columns


def build_raw() -> tuple[pd.DataFrame, list[str]]:
    weather = parse_epw(EPW)
    energy, cooling_columns = parse_cooling(SIM_CSV)
    if not weather[["month", "day", "hour"]].reset_index(drop=True).equals(
        energy[["month", "day", "hour"]].reset_index(drop=True)
    ):
        raise ValueError("weather and energy timestamps do not align")

    merged = pd.concat(
        [weather.reset_index(drop=True),
         energy[[
             "day_of_week", "day_of_year", "is_weekend", TARGET,
             "building_cooling_energy_w", "datacenter_cooling_energy_w",
         ]].reset_index(drop=True)],
        axis=1,
    )
    merged["scenario"] = SCENARIO
    merged["scenario_row"] = np.arange(len(merged))
    merged["lag_1"] = merged[TARGET].shift(1)
    merged["lag_24"] = merged[TARGET].shift(24)
    before = len(merged)
    merged = merged.dropna(subset=[TARGET, "lag_1", "lag_24"]).reset_index(drop=True)
    merged["sequence_index"] = np.arange(len(merged))

    output_columns = (["sequence_index", "scenario", "scenario_row"] + FEATURES
                      + ["building_cooling_energy_w", "datacenter_cooling_energy_w", TARGET])
    raw = merged[output_columns].copy()
    raw.to_csv(RAW_OUT, index=False, encoding="utf-8")
    return raw, cooling_columns


def build_platform(raw: pd.DataFrame) -> pd.DataFrame:
    """Adapt the raw frame into the platform upload layout."""

    month = raw["month"].astype("int64")
    day = raw["day"].astype("int64")
    hour = raw["hour"].astype("int64")
    calendar_dates = pd.to_datetime(
        pd.DataFrame({
            "year": pd.Series(REFERENCE_YEAR, index=raw.index),
            "month": month, "day": day,
        }),
        errors="coerce",
    )
    if calendar_dates.isna().any():
        raise ValueError("month/day contains invalid dates")
    # Hours are 1-based: hour=24 belongs to the following day at 00:00.
    local_hours = (
        (calendar_dates - pd.Timestamp(f"{REFERENCE_YEAR}-01-01")) / pd.Timedelta(hours=1)
    ).astype("int64") + hour
    scenario_code = pd.Series(0, index=raw.index, dtype="int64")
    scenario_span = int(local_hours.max()) + 1
    platform_key = scenario_code * scenario_span + local_hours
    if len(np.unique(platform_key)) != len(platform_key):
        raise ValueError("derived platform timeline contains duplicates")
    timestamps = PLATFORM_EPOCH + pd.to_timedelta(platform_key, unit="h")

    output = pd.DataFrame(index=raw.index)
    output["timestamp"] = timestamps
    output["sequence_index"] = raw["sequence_index"].to_numpy()
    output["scenario_code"] = scenario_code.to_numpy()
    output["scenario_row"] = raw["scenario_row"].to_numpy()
    for column in [
        "month", "day", "hour", "day_of_week", "day_of_year", "is_weekend",
        "dry_bulb_temp_c", "dew_point_temp_c", "relative_humidity_pct", "pressure_pa",
        "global_horizontal_radiation_wh_m2", "direct_normal_radiation_wh_m2",
        "diffuse_horizontal_radiation_wh_m2", "wind_direction_deg", "wind_speed_m_s",
        "total_sky_cover_tenths", "opaque_sky_cover_tenths", "snow_depth_cm",
        "lag_1", "lag_24",
    ]:
        output[column] = raw[column].to_numpy()
    output[TARGET] = raw[TARGET].to_numpy()
    output = output.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    parsed = pd.to_datetime(output["timestamp"], errors="coerce", format="mixed")
    if parsed.isna().any() or not parsed.is_unique or not parsed.is_monotonic_increasing:
        raise ValueError("platform timestamp must be parseable, unique and increasing")
    if not np.isfinite(output[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("target must be finite")
    output.to_csv(PLATFORM_OUT, index=False, encoding="utf-8")
    return output


def _configure(args: argparse.Namespace) -> None:
    global EPW, SIM_CSV, RAW_OUT, PLATFORM_OUT, SUMMARY_OUT
    EPW = args.epw
    SIM_CSV = args.sim_csv
    RAW_OUT = args.raw_output
    PLATFORM_OUT = args.platform_output
    SUMMARY_OUT = args.summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epw", type=Path, default=DEFAULT_EPW)
    parser.add_argument("--sim-csv", type=Path, default=DEFAULT_SIM_CSV)
    parser.add_argument("--raw-output", type=Path, default=RAW_OUT)
    parser.add_argument("--platform-output", type=Path, default=PLATFORM_OUT)
    parser.add_argument("--summary", type=Path, default=SUMMARY_OUT)
    _configure(parser.parse_args(argv))
    for directory in {RAW_OUT.parent, PLATFORM_OUT.parent, SUMMARY_OUT.parent}:
        directory.mkdir(parents=True, exist_ok=True)
    raw, cooling_columns = build_raw()
    platform = build_platform(raw)
    summary = {
        "weather_source": str(EPW),
        "energy_source": str(SIM_CSV),
        "location": "Anqing, Anhui, China (WMO 584240)",
        "building_model": "ASHRAE901 OfficeLarge STD2019 Chiller205 Detailed",
        "cooling_columns": cooling_columns,
        "cooling_column_count": len(cooling_columns),
        "target": TARGET,
        "target_unit": "W",
        "target_definition": "sum of all Air System Total Cooling Energy [J] / 3600 s",
        "raw_output": str(RAW_OUT),
        "platform_output": str(PLATFORM_OUT),
        "raw_rows": len(raw),
        "platform_rows": len(platform),
        "feature_count": len(FEATURES),
        "feature_columns": FEATURES,
        "lag_policy": "lag_1 and lag_24 generated from the ordered target series",
        "timestamp_note": "derived platform ordering key; not a real historical year",
        "target_stats": {
            "min": float(raw[TARGET].min()),
            "max": float(raw[TARGET].max()),
            "mean": float(raw[TARGET].mean()),
            "zero_hours": int((raw[TARGET] == 0).sum()),
        },
        "target_composition": {
            "building_loops": [
                c for c in cooling_columns if "DATACENTER" not in c.upper()
            ],
            "datacenter_loops": [
                c for c in cooling_columns if "DATACENTER" in c.upper()
            ],
            "building_mean_w": float(raw["building_cooling_energy_w"].mean()),
            "datacenter_mean_w": float(raw["datacenter_cooling_energy_w"].mean()),
            "building_share_pct": float(
                raw["building_cooling_energy_w"].sum() / raw[TARGET].sum() * 100
            ),
            "note": (
                "target_total_cooling_energy_w sums every air-loop cooling coil; "
                "building_cooling_energy_w excludes the four DATACENTER loops, "
                "which run as a near-constant baseload"
            ),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
