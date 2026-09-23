.PHONY: frontend-data evaluation-report eda gru-seed-robustness provenance reproduce-track-only serve test

frontend-data:
	python3 -m src.export_replay

evaluation-report:
	python3 -m src.report_evaluation

eda:
	test -n "$(IBTRACS_CSV)"
	python3 -m src.report_eda "$(IBTRACS_CSV)"

gru-seed-robustness:
	test -n "$(IBTRACS_CSV)"
	KMP_DUPLICATE_LIB_OK=TRUE python3 -m src.run_gru_seed_robustness "$(IBTRACS_CSV)"

provenance:
	test -n "$(IBTRACS_CSV)"
	python3 -m src.write_provenance "$(IBTRACS_CSV)"

reproduce-track-only:
	test -n "$(IBTRACS_CSV)"
	KMP_DUPLICATE_LIB_OK=TRUE python3 -m src.run_baselines "$(IBTRACS_CSV)" --output results/track_only_baselines.csv
	KMP_DUPLICATE_LIB_OK=TRUE python3 -m src.run_gru "$(IBTRACS_CSV)" --output results/track_only_gru_comparison.csv
	KMP_DUPLICATE_LIB_OK=TRUE python3 -m src.run_error_analysis "$(IBTRACS_CSV)" --output-dir results/track_only_error_analysis
	python3 -m src.report_evaluation
	python3 -m src.report_eda "$(IBTRACS_CSV)"
	KMP_DUPLICATE_LIB_OK=TRUE python3 -m src.run_gru_seed_robustness "$(IBTRACS_CSV)"
	python3 -m src.write_provenance "$(IBTRACS_CSV)"

serve: frontend-data
	python3 -m http.server 8000 --directory frontend

test:
	KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest discover -s tests -p 'test_*.py' -v
