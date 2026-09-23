PYTHON ?= python3
PORT ?= 8767

.PHONY: serve check test verify fetch rebuild pipeline

serve:
	$(PYTHON) -m http.server $(PORT) --bind 127.0.0.1 --directory app

check:
	$(PYTHON) -m py_compile src/*.py

test:
	$(PYTHON) -m unittest discover -s tests -v

verify: check test

fetch:
	$(PYTHON) src/fetch_data.py

rebuild: check
	$(PYTHON) src/build_features.py
	$(PYTHON) src/target_model_experiments.py
	$(PYTHON) src/feature_experiments.py
	$(PYTHON) src/model_weekly_passers.py
	$(PYTHON) src/generate_dashboard.py

pipeline: fetch rebuild
