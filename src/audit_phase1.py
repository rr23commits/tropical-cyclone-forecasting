#!/usr/bin/env python3
"""Report Phase 1 sample counts from an official IBTrACS North Atlantic CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

from ibtracs import build_forecast_samples, filter_north_atlantic, load_ibtracs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=Path, help="official ibtracs.NA.list.v04r01.csv")
    args = parser.parse_args()

    filtered = filter_north_atlantic(load_ibtracs(args.csv))
    print(f"Filtered six-hour states: {len(filtered):,} | storms: {filtered.SID.nunique():,}")
    for horizon in (6, 24, 48):
        samples = build_forecast_samples(filtered, horizon)
        print(
            f"+{horizon:>2}h: storms: {samples.metadata.SID.nunique():,} "
            f"| forecast origins: {len(samples.metadata):,}"
        )


if __name__ == "__main__":
    main()
