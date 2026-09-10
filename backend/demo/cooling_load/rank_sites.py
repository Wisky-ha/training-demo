"""Rank pre-computed simulations by how closely their cooling profile matches Anqing.

This is the auditable basis for the site selection hard-coded in
``build_climate_years.py``:

    combined = normalised monthly shape difference
             + relative annual cooling difference

Usage::

    python backend/demo/cooling_load/rank_sites.py --output ranking.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

DEFAULT_SIM_DIR = Path(r"C:/EnergyPlusV26-1-0/forAgent/simulation_all")
DEFAULT_EPW_DIR = Path(r"C:/EnergyPlusV26-1-0/forAgent/epw_flat")
REFERENCE = "CHN_AH_Anqing.584240_TMYx.2011-2025"


def read_cooling(path: Path) -> np.ndarray | None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        header = [c.strip() for c in next(reader)]
        index = [i for i, name in enumerate(header) if "Total Cooling Energy" in name]
        if not index:
            return None
        values = []
        for row in reader:
            if len(row) <= max(index):
                continue
            try:
                values.append(sum(float(row[i]) for i in index) / 3600.0)
            except ValueError:
                return None
    if len(values) != 8760:
        return None
    return np.asarray(values, dtype=float)


def profile(series: np.ndarray, month_index: np.ndarray) -> dict:
    monthly = [float(series[month_index == m].mean()) for m in range(1, 13)]
    return {
        "annual_kwh": float(series.sum() / 1000.0),
        "mean_w": float(series.mean()),
        "peak_w": float(series.max()),
        "min_w": float(series.min()),
        "monthly_mean_w": monthly,
        "load_factor": float(series.mean() / series.max()) if series.max() else None,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", type=Path, default=DEFAULT_SIM_DIR)
    parser.add_argument("--epw-dir", type=Path, default=DEFAULT_EPW_DIR)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--top", type=int, default=40)
    args = parser.parse_args(argv)
    SIM_DIR = args.sim_dir
    EPW_FLAT = args.epw_dir
    OUT = args.output

    month_index = np.repeat(np.arange(1, 13), [744, 672, 744, 720, 744, 720, 744, 744, 720, 744, 720, 744])[:8760]
    assert len(month_index) == 8760

    entries = []
    skipped = []
    for csv_path in sorted(SIM_DIR.glob("*.csv")):
        stem = csv_path.stem
        series = read_cooling(csv_path)
        if series is None:
            skipped.append(stem)
            continue
        entries.append({"stem": stem, **profile(series, month_index)})
        entries[-1]["_series"] = series

    reference = next(e for e in entries if e["stem"] == REFERENCE)
    ref_monthly = np.asarray(reference["monthly_mean_w"])
    ref_annual = reference["annual_kwh"]

    for entry in entries:
        monthly = np.asarray(entry["monthly_mean_w"])
        # Shape distance: normalised monthly profile difference.
        shape = float(np.sqrt(np.mean(
            ((monthly / monthly.mean()) - (ref_monthly / ref_monthly.mean())) ** 2
        )))
        # Magnitude distance: relative annual energy difference.
        magnitude = float(abs(entry["annual_kwh"] - ref_annual) / ref_annual)
        entry["shape_distance"] = shape
        entry["magnitude_distance"] = magnitude
        entry["combined"] = shape + magnitude
        entry.pop("_series")

    entries.sort(key=lambda e: e["combined"])
    for entry in entries:
        entry["epw_exists"] = (EPW_FLAT / f"{entry['stem']}.epw").exists()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "reference": {k: v for k, v in reference.items() if k != "_series"},
        "profiled": len(entries),
        "skipped": skipped,
        "ranking": entries,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"profiled={len(entries)} skipped={len(skipped)}")
    print(f"reference {REFERENCE}: annual={ref_annual/1000:,.0f} MWh mean={reference['mean_w']:,.0f} W")
    print(f"{'#':>3} {'site':<52} {'annual MWh':>10} {'mean W':>9} {'shape':>6} {'magn':>6} {'comb':>6}")
    for i, e in enumerate(entries[:args.top], 1):
        print(f"{i:>3} {e['stem'][:51]:<52} {e['annual_kwh']/1000:>10,.0f} {e['mean_w']:>9,.0f} "
              f"{e['shape_distance']:>6.3f} {e['magnitude_distance']:>6.3f} {e['combined']:>6.3f}")


if __name__ == "__main__":
    main()
