#!/usr/bin/env python3
"""Summarize one or more official IBTrACS basin CSVs by storm.

Download the CSVs outside this repository, then run:
  python src/audit_ibtracs.py /path/to/ibtracs.NA.list.v04r01.csv \
    /path/to/ibtracs.WP.list.v04r01.csv

Requires: pandas.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


CORE = [
    "SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "TRACK_TYPE", "LAT", "LON",
    "WMO_WIND", "WMO_PRES", "WMO_AGENCY", "USA_LAT", "USA_LON", "USA_WIND",
    "USA_PRES", "USA_AGENCY", "USA_STATUS", "IFLAG",
]
TROPICAL = {"TD", "TS", "HU", "TY", "ST", "TC", "SD", "SS"}
HORIZONS = (6, 24, 48)


def load(path: Path) -> pd.DataFrame:
    """The second CSV row is units, not an observation."""
    # `NA` is a valid basin code, so pandas must not treat it as a missing value.
    data = pd.read_csv(
        path, skiprows=[1], usecols=lambda column: column in CORE, low_memory=False,
        keep_default_na=False, na_values=[" "],
    )
    missing = set(CORE) - set(data.columns)
    assert not missing, f"{path} is missing {sorted(missing)}"
    data["ISO_TIME"] = pd.to_datetime(data["ISO_TIME"], errors="coerce")
    data["SEASON"] = pd.to_numeric(data["SEASON"], errors="coerce")
    for column in ("LAT", "LON", "WMO_WIND", "WMO_PRES", "USA_LAT", "USA_LON", "USA_WIND", "USA_PRES"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    return data


def prepared(data: pd.DataFrame, basin: str) -> pd.DataFrame:
    """Use completed seasons, main tracks, and actual six-hour forecast issue times."""
    data = data[
        (data["SEASON"].between(1980, 2025))
        & (data["TRACK_TYPE"] == "main")
        & (data["BASIN"] == basin)
    ].copy()
    data = data.dropna(subset=["SID", "ISO_TIME"])
    data["six_hour"] = (data["ISO_TIME"].dt.minute == 0) & (data["ISO_TIME"].dt.hour % 6 == 0)
    data["complete_state"] = data[["USA_LAT", "USA_LON", "USA_WIND"]].notna().all(axis=1)
    data["tropical"] = data["USA_STATUS"].isin(TROPICAL)
    data["usa_iflag"] = data["IFLAG"].fillna("").str[0]
    return data


def horizon_counts(data: pd.DataFrame, include_pressure: bool = False) -> dict[int, tuple[int, int]]:
    """Count storms and origins with a complete tropical US-agency state at t and t+h."""
    usable = data[data["six_hour"] & data["complete_state"] & data["tropical"]].copy()
    if include_pressure:
        usable = usable[usable["USA_PRES"].notna()]
    usable = usable.drop_duplicates(["SID", "ISO_TIME"], keep="first")
    indexed = usable.set_index(["SID", "ISO_TIME"])
    counts = {}
    for horizon in HORIZONS:
        future = indexed.copy()
        future.index = pd.MultiIndex.from_arrays(
            [future.index.get_level_values("SID"), future.index.get_level_values("ISO_TIME") - pd.Timedelta(hours=horizon)],
            names=["SID", "ISO_TIME"],
        )
        matches = indexed.index.intersection(future.index)
        counts[horizon] = (len(matches.get_level_values("SID").unique()), len(matches))
    return counts


def describe(data: pd.DataFrame, label: str) -> None:
    storms = data["SID"].nunique()
    obs = len(data)
    duplicates = data.duplicated(["SID", "ISO_TIME"]).sum()
    six = data[data["six_hour"]]
    core = six["complete_state"]
    rates = six[["USA_LAT", "USA_LON", "USA_WIND", "USA_PRES"]].isna().mean().mul(100)
    intervals = (
        data.sort_values(["SID", "ISO_TIME"]).groupby("SID")["ISO_TIME"].diff().dt.total_seconds().div(3600)
    )
    interval_counts = intervals.value_counts().head(12)
    primary_agency = six.loc[core, "USA_AGENCY"].dropna().mode().iat[0]
    strict = data[data["USA_AGENCY"] == primary_agency]

    print(f"\n## {label}")
    print(f"Completed main-track seasons: 1980–2025 | storms: {storms:,} | observations: {obs:,}")
    print(f"Year range present: {int(data.SEASON.min())}–{int(data.SEASON.max())} | duplicate (SID, timestamp) rows: {duplicates:,}")
    print(f"Six-hour ticks: {len(six):,} | complete USA position+wind states: {core.sum():,} ({core.mean() * 100:.1f}%)")
    print("Six-hour USA missing rates (%): " + ", ".join(f"{name[4:].lower()}={value:.1f}" for name, value in rates.items()))
    merged_rates = six[["LAT", "LON", "WMO_WIND", "WMO_PRES"]].isna().mean().mul(100)
    print("Six-hour merged/WMO missing rates (%): " + ", ".join(f"{name.lower()}={value:.1f}" for name, value in merged_rates.items()))
    print("Most common raw intervals (hours): " + ", ".join(f"{hours:g}={count:,}" for hours, count in interval_counts.items()))
    print("USA agency values on complete six-hour states:")
    print(six.loc[core, "USA_AGENCY"].fillna("<missing>").value_counts().head(10).to_string())
    print("USA IFLAG first-character values on complete six-hour states:")
    print(six.loc[core, "usa_iflag"].replace("", "<missing>").value_counts().to_string())
    print("Usable tropical storms / valid forecast origins:")
    for horizon, (storm_count, origin_count) in horizon_counts(data).items():
        print(f"  +{horizon:>2}h: {storm_count:,} / {origin_count:,}")
    print(f"Strict primary USA source ({primary_agency}) storms / valid origins:")
    for horizon, (storm_count, origin_count) in horizon_counts(strict).items():
        print(f"  +{horizon:>2}h: {storm_count:,} / {origin_count:,}")
    print("Strict primary USA source, also requiring pressure at issue and target:")
    for horizon, (storm_count, origin_count) in horizon_counts(strict, include_pressure=True).items():
        print(f"  +{horizon:>2}h: {storm_count:,} / {origin_count:,}")
    print("Storms by year:")
    by_year = data.groupby("SEASON")["SID"].nunique()
    print(by_year.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path, help="official IBTrACS basin CSV file(s)")
    args = parser.parse_args()
    for path in args.csv:
        basin = path.name.split(".")[1].upper()
        data = prepared(load(path), basin)
        assert not data.empty, f"no 1980–2025 main-track observations in {path}"
        describe(data, path.name)


if __name__ == "__main__":
    main()
