# South Asian LPS–TCC metadata crosswalk

- TCC rows examined: 1276; zero/partial/full matches: 1275/1/0.
- Full 9/9 coverage: 0.000%.
- The source Parquet has 368,925 rows, 276 columns, and spans 1940-05-17 through 2025-12-03 (291,600,496 bytes). It contains rows at 3,120/6,438 required TCC timestamps (48.46%).
- Accepted unique-position distances (km): {'count': 5, 'median': 94.07417415221211, 'mean': 89.76395619861941, 'p90': 108.14118916753456, 'maximum': 110.26836159234752}.
- Ambiguous TCC positions: 0; non-detected matches used: 0.
- Unique LPS tracks used: 1; full rows with both features complete: 0.
- The one partial TCC row matched five positions: vorticity is non-null at 5/5; RH is non-null at 0/5. RH is null in all 368,925 source rows; consequently, there are no complete 9/9 matched rows for both predictors.
- The sole partial candidate is TCC track 103109 (NI, 2013, test, label_24h=0), with 5/9 timestamps matched to LPS track 6846. Strict retained subset: 0 rows.

## Full-match rates

| Grouping | Group | Rows | Full 9/9 | Percent |
|---|---:|---:|---:|---:|
| Basin | EP | 439 | 0 | 0.000% |
| Basin | NA | 397 | 0 | 0.000% |
| Basin | NI | 45 | 0 | 0.000% |
| Basin | SA | 1 | 0 | 0.000% |
| Basin | SI | 124 | 0 | 0.000% |
| Basin | SP | 54 | 0 | 0.000% |
| Basin | WP | 216 | 0 | 0.000% |
| Split | test | 331 | 0 | 0.000% |
| Split | train | 804 | 0 | 0.000% |
| Split | validation | 141 | 0 | 0.000% |
| 24h label | 0 | 437 | 0 | 0.000% |
| 24h label | 1 | 839 | 0 | 0.000% |

## Retained subset

Full-match rows: 0; unique TCC tracks: 0; years: [].
Frozen split/label counts: {}.
Semester-project screening rule: at least 100 full rows, 20 unique TCC tracks, 20 rows in each label, at least 10 of each label in every frozen split, both predictors complete on at least 90% of full rows, and no LPS track used by more than 10% of full rows.
LPS intensity/category fields were not used as predictors or labels. TCC labels and splits are unchanged.

**C — crosswalk fails; abandon this catalogue for TCC enrichment**
