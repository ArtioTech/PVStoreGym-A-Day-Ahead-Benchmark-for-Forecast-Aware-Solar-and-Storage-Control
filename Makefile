# One command per result. PYTHON can be overridden: make test PYTHON=.venv/bin/python
PYTHON ?= .venv/bin/python
# PPO-Lagrangian (OmniSafe) and the rliable statistics need Python 3.9/3.10: requirements-omnisafe.lock
PYTHON_OMNISAFE ?= .venv-omnisafe/bin/python

.PHONY: help data test conformance audit benchmark rl summary compute figures

help:
	@echo "make data         build the dataset and the publication-time files from the publishers' sources"
	@echo "make test         run all tests -> results/<subdir>/tests.json"
	@echo "make conformance  settlement conformance on 5 days -> results/<subdir>/conformance.json"
	@echo "make audit        conference configuration under settlement v1 and v2 -> results/audit.json"
	@echo "make benchmark    oracle + optimisation baselines + RL + summaries -> results/<subdir>/benchmark_<task>.csv, summary_<task>.json"
	@echo "make compute      timing measurements -> results/<subdir>/compute.json"
	@echo "make figures      figures/<subdir>/*.pdf from the saved results (subdir = results_subdir of PROTOCOL, rev1 by default)"

data:
	$(PYTHON) scripts/auxiliary/build_greek_2025_dataset.py --year 2025
	$(PYTHON) scripts/data/build_publication_sidecars.py --year 2025
	$(PYTHON) scripts/data/verify_dataset.py data/processed

test:
	$(PYTHON) scripts/make_tests_json.py

conformance:
	$(PYTHON) scripts/validation/validate_settlement.py

audit:
	$(PYTHON) scripts/run_audit.py

# T4/T5 read c_thr (5.7 EUR/MWh) from their task files; C_THR=<EUR/MWh> overrides it for a sensitivity run.
TASKS ?= T1-Nominal T2-Forecast T3-Gate T4-Ageing T5-Combined
C_THR_ARG = $(if $(C_THR),--c-thr $(C_THR),)

# Protocol: configs/journal.yaml (rev1, results/rev1/); v3: PROTOCOL=configs/journal_v3.yaml; v4: PROTOCOL=configs/journal_v4.yaml
PROTOCOL ?= configs/journal.yaml
export PVBESS_PROTOCOL = $(PROTOCOL)

benchmark:
	$(PYTHON) scripts/run_journal.py oracle --tasks $(TASKS) $(C_THR_ARG)
	$(if $(findstring journal_v4,$(PROTOCOL)),,$(PYTHON) scripts/run_journal.py oracle_deliver --tasks $(TASKS) $(C_THR_ARG))
	$(PYTHON) scripts/run_journal.py baselines --tasks $(TASKS) $(C_THR_ARG)
	$(PYTHON) scripts/run_journal.py aggregate --tasks $(TASKS) $(C_THR_ARG)
	$(MAKE) rl summary

rl:
	$(PYTHON) scripts/rl/run_all.py tune --python $(PYTHON) --python-omnisafe $(PYTHON_OMNISAFE)
	$(PYTHON) scripts/rl/run_all.py main --tasks $(TASKS) $(C_THR_ARG) --python $(PYTHON) --python-omnisafe $(PYTHON_OMNISAFE)
	$(PYTHON) scripts/rl/collect_rl.py

summary:
	$(PYTHON) scripts/run_journal.py aggregate --tasks $(TASKS) $(C_THR_ARG)
	$(PYTHON_OMNISAFE) scripts/summarise_journal.py --tasks $(TASKS)

compute:
	$(PYTHON) scripts/measure_compute.py

figures:
	for f in fig_scores_T1 fig_decomposition_T1 fig_tasks fig_learning; do $(PYTHON) figures/src/$$f.py || exit 1; done
