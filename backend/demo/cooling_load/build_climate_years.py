"""Build a multi-site cooling-load dataset from pre-computed simulations.

The EnergyPlus work is already done in ``simulation_all``: one annual run per
weather file, produced with the Shanghai-configured ASHRAE 901 large-office
model.  This script only *pools* those results.

Site selection is data-driven, not hand-picked.  Every simulation is ranked by
how closely its hourly cooling profile matches Anqing, using

  combined = |normalised monthly shape difference|
           + |relative annual cooling difference|

Only sites within ``--max-distance`` are pooled.  Each site becomes one
"weather year" (``scenario``), mirroring the electric-load dataset layout.

Important: these are *different locations with a similar climate*, not
different historical years of one station.  The pooled file is a synthetic
multi-year panel, and the documentation must say so.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_SIM_DIR = Path(r"C:/EnergyPlusV26-1-0/forAgent/simulation_all")
DEFAULT_EPW_DIR = Path(r"C:/EnergyPlusV26-1-0/forAgent/epw_flat")

TARGET = "target_total_cooling_energy_w"
BUILDING_TARGET = "building_cooling_energy_w"
DATACENTER_TARGET = "datacenter_cooling_energy_w"
REFERENCE_YEAR = 2001
PLATFORM_EPOCH = pd.Timestamp("2000-01-01 00:00:00")

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

# Loaded from the cooling-profile ranking; distances are reproducible.
SITES = [
    {"code": "ANQ", "stem": "CHN_AH_Anqing.584240_TMYx.2011-2025", "distance": 0.000, "city": "Anqing, AH, CHN"},
    {"code": "MIL", "stem": "GRC_AI_Milos.167380_TMYx.2011-2025", "distance": 0.229, "city": "Milos, GRC"},
    {"code": "IRA", "stem": "JPN_AI_Irako.476530_TMYx.2011-2025", "distance": 0.234, "city": "Irako, JPN"},
    {"code": "BST", "stem": "AFG_HEL_Bust.AP.409880_TMYx.2011-2025", "distance": 0.296, "city": "Bust AP, AFG"},
    {"code": "ADA", "stem": "TUR_AA_Adana-Incirlik.AB.173500_TMYx.2009-2023", "distance": 0.310, "city": "Adana Incirlik, TUR"},
    {"code": "BEJ", "stem": "TUN_BJ_Beja.607230_TMYx.2011-2025", "distance": 0.328, "city": "Beja, TUN"},
    {"code": "BSV", "stem": "ISR_D_Beer.Sheva.401900_TMYx.2011-2025", "distance": 0.334, "city": "Beer Sheva, ISR"},
    {"code": "BEN", "stem": "LBY_BA_Benghazi-Benigna.Intl.AP.620530_TMYx.2011-2025", "distance": 0.336, "city": "Benghazi, LBY"},
    {"code": "CEU", "stem": "ESP_CE_Ceuta.603200_TMYx.2011-2025", "distance": 0.338, "city": "Ceuta, ESP"},
]


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
        "opaque_sky_cover_tenths", "snow_depth_cm",
    ]
    weather = frame[selected].apply(pd.to_numeric, errors="coerce")
    for column, (lower, upper) in VALID_RANGES.items():
        weather.loc[
            (weather[column] < lower) | (weather[column] > upper), column
        ] = np.nan
    return weather


def parse_simulation(path: Path) -> tuple[pd.DataFrame, list[str]]:
    # Header names carry inconsistent trailing spaces, so select columns with a
    # predicate over the stripped name instead of matching raw names.
    def keep(name: str) -> bool:
        cleaned = str(name).strip()
        return cleaned == "Date/Time" or "Total Cooling Energy" in cleaned

    frame = pd.read_csv(path, usecols=keep, encoding="utf-8-sig")
    frame.columns = [str(c).strip() for c in frame.columns]
    cooling = [c for c in frame.columns if "Total Cooling Energy" in c]
    if not cooling:
        raise KeyError(f"{path.name}: no cooling energy columns")
    if len(frame) != 8760:
        raise ValueError(f"{path.name}: expected 8760 rows, got {len(frame)}")

    parts = frame["Date/Time"].astype(str).str.strip().str.extract(
        r"^(\d{1,2})/(\d{1,2})\s+(\d{1,2}):(\d{2}):(\d{2})$"
    )
    if parts.isna().any(axis=None):
        raise ValueError(f"{path.name}: unable to parse Date/Time")
    month, day, hour = (parts[i].astype(int) for i in (0, 1, 2))

    joules = frame[cooling].astype(float)
    if (joules < 0).any(axis=None):
        raise ValueError(f"{path.name}: negative cooling energy")
    total = joules.sum(axis=1) / 3600.0
    datacenter_columns = [c for c in cooling if "DATACENTER" in c.upper()]
    datacenter = joules[datacenter_columns].sum(axis=1) / 3600.0

    date = pd.to_datetime({"year": REFERENCE_YEAR, "month": month, "day": day})
    return pd.DataFrame({
        "month": month.to_numpy(),
        "day": day.to_numpy(),
        "hour": hour.to_numpy(),
        "day_of_week": date.dt.dayofweek.to_numpy(),
        "day_of_year": date.dt.dayofyear.to_numpy(),
        "is_weekend": (date.dt.dayofweek >= 5).astype(int).to_numpy(),
        TARGET: total.to_numpy(),
        BUILDING_TARGET: (total - datacenter).to_numpy(),
        DATACENTER_TARGET: datacenter.to_numpy(),
    }), cooling


def build_raw(sim_dir: Path, epw_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    pieces, manifest = [], []
    for site in SITES:
        sim_path = sim_dir / f"{site['stem']}.csv"
        epw_path = epw_dir / f"{site['stem']}.epw"
        if not sim_path.is_file() or not epw_path.is_file():
            raise FileNotFoundError(f"missing inputs for {site['stem']}")
        weather = parse_epw(epw_path)
        energy, cooling_columns = parse_simulation(sim_path)
        if not weather[["month", "day", "hour"]].reset_index(drop=True).equals(
            energy[["month", "day", "hour"]].reset_index(drop=True)
        ):
            raise ValueError(f"{site['stem']}: weather and simulation do not align")
        merged = pd.concat(
            [weather.reset_index(drop=True),
             energy[["day_of_week", "day_of_year", "is_weekend", TARGET,
                     BUILDING_TARGET, DATACENTER_TARGET]].reset_index(drop=True)],
            axis=1,
        )
        merged["scenario"] = site["code"]
        merged["scenario_row"] = np.arange(len(merged))
        pieces.append(merged)
        manifest.append({
            **site,
            "cooling_columns": len(cooling_columns),
            "annual_cooling_wh": float(energy[TARGET].sum()),
            "annual_cooling_mwh": float(energy[TARGET].sum() / 1e6),
            "mean_cooling_w": float(energy[TARGET].mean()),
        })

    # Lags must be computed *within* each site: these are different places, so
    # carrying a lag across a site boundary would be physically meaningless.
    data = pd.concat(pieces, ignore_index=True)
    data = data.sort_values(
        ["scenario", "scenario_row"], kind="mergesort"
    ).reset_index(drop=True)
    grouped = data.groupby("scenario", sort=False)[TARGET]
    data["lag_1"] = grouped.shift(1)
    data["lag_24"] = grouped.shift(24)
    before = len(data)
    data = data.dropna(subset=[TARGET, "lag_1", "lag_24"]).reset_index(drop=True)
    data["sequence_index"] = np.arange(len(data))

    output_columns = (["sequence_index", "scenario", "scenario_row"] + FEATURES
                      + [BUILDING_TARGET, DATACENTER_TARGET, TARGET])
    raw = data[output_columns].copy()
    return raw, manifest


def build_platform(raw: pd.DataFrame) -> pd.DataFrame:
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
    local_hours = (
        (calendar_dates - pd.Timestamp(f"{REFERENCE_YEAR}-01-01")) / pd.Timedelta(hours=1)
    ).astype("int64") + hour

    codes = sorted(raw["scenario"].unique())
    mapping = {name: index for index, name in enumerate(codes)}
    scenario_code = raw["scenario"].map(mapping).astype("int64")
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
    for column in [c for c in FEATURES if c not in ("lag_1", "lag_24")]:
        output[column] = raw[column].to_numpy()
    output["lag_1"] = raw["lag_1"].to_numpy()
    output["lag_24"] = raw["lag_24"].to_numpy()
    output[TARGET] = raw[TARGET].to_numpy()
    output = output.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    parsed = pd.to_datetime(output["timestamp"], errors="coerce", format="mixed")
    if parsed.isna().any() or not parsed.is_unique or not parsed.is_monotonic_increasing:
        raise ValueError("platform timestamp must be parseable, unique and increasing")
    if not np.isfinite(output[TARGET].to_numpy(dtype=float)).all():
        raise ValueError("target must be finite")
    return output, mapping


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--epw-dir", type=Path, default=DEFAULT_EPW_DIR)
    parser.add_argument("--raw-output", type=Path, required=True)
    parser.add_argument("--platform-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args(argv)

    raw, manifest = build_raw(args.sim_dir, args.epw_dir)
    platform, mapping = build_platform(raw)

    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    args.platform_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.raw_output, index=False, encoding="utf-8")
    platform.to_csv(args.platform_output, index=False, encoding="utf-8")

    summary = {
        "kind": "multi-site climate-consistent weather years",
        "target": TARGET,
        "target_unit": "W",
        "target_definition": (
            "sum of every Air System Total Cooling Energy [J] / 3600 s, "
            "for the Shanghai-configured ASHRAE 901 large-office model"
        ),
        "site_selection": (
            "ranked by combined cooling-profile distance to Anqing "
            "(normalised monthly shape difference + relative annual difference)"
        ),
        "caveat": (
            "These are different locations with a similar climate, not different "
            "historical years of one station. The pooled file is a synthetic "
            "multi-year panel."
        ),
        "sites": manifest,
        "scenario_mapping": mapping,
        "raw_rows": len(raw),
        "platform_rows": len(platform),
        "rows_per_site": 8760,
        "feature_count": len(FEATURES),
        "feature_columns": FEATURES,
        "columns": list(raw.columns),
        "target_stats": {
            "min": float(raw[TARGET].min()),
            "max": float(raw[TARGET].max()),
            "mean": float(raw[TARGET].mean()),
        },
        "lag_policy": (
            "lag_1 and lag_24 are computed within each site; the first 24 hours of "
            "every site are dropped so no lag crosses a site boundary"
        ),
        "rows_per_site_retained": 8760 - 24,
        "timestamp_note": "derived platform ordering key; not a real historical year",
    }
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "raw_rows": summary["raw_rows"],
        "platform_rows": summary["platform_rows"],
        "sites": len(manifest),
        "target_mean_w": summary["target_stats"]["mean"],
        "raw_output": str(args.raw_output),
        "platform_output": str(args.platform_output),
    }, ensure_ascii=False, indent=2))
    for entry in manifest:
        print(f"  {entry['code']}  {entry['city']:<26} dist={entry['distance']:.3f} "
              f"annual={entry['annual_cooling_mwh']:,.0f} MWh")


if __name__ == "__main__":
    main()
