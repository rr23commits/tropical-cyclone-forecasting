import unittest

import pandas as pd

from src.ibtracs import filter_north_atlantic
from src.report_eda import quality_tables


class EDAReportTests(unittest.TestCase):
    def test_quality_tables_keep_missingness_and_eligibility_separate(self):
        raw = pd.DataFrame([
            {"SID": "A", "SEASON": 2000, "NAME": "A", "BASIN": "NA", "ISO_TIME": "2000-08-01T00:00:00Z", "TRACK_TYPE": "main", "USA_LAT": 10, "USA_LON": -50, "USA_WIND": 40, "USA_PRES": None, "USA_AGENCY": "hurdat_atl", "USA_STATUS": "TS", "IFLAG": "O"},
            {"SID": "B", "SEASON": 2000, "NAME": "B", "BASIN": "NA", "ISO_TIME": "2000-08-01T06:00:00Z", "TRACK_TYPE": "main", "USA_LAT": 10, "USA_LON": -50, "USA_WIND": None, "USA_PRES": 1000, "USA_AGENCY": "hurdat_atl", "USA_STATUS": "TS", "IFLAG": "O"},
        ])
        raw["ISO_TIME"] = pd.to_datetime(raw["ISO_TIME"], utc=True)
        counts, missing = quality_tables(raw, filter_north_atlantic(raw), 2015, 2019)
        self.assertEqual(counts.loc[counts.stage.eq("complete position and wind"), "states"].item(), 1)
        self.assertEqual(missing.loc[missing.field.eq("USA_WIND"), "missing_percent"].item(), 50.0)
        self.assertEqual(missing.loc[missing.field.eq("USA_PRES"), "missing_percent"].item(), 50.0)
