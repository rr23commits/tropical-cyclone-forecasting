.PHONY: frontend-data evaluation-report serve test

frontend-data:
	python3 -m src.export_replay

evaluation-report:
	python3 -m src.report_evaluation

serve: frontend-data
	python3 -m http.server 8000 --directory frontend

test:
	KMP_DUPLICATE_LIB_OK=TRUE python3 -m unittest discover -s tests -p 'test_*.py' -v
