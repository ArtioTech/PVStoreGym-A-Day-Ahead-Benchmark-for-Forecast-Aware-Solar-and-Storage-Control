# Installation

Requires Python 3.10 or newer. Tested on Python 3.10, 3.12 and 3.14 (macOS
arm64, Linux x86-64). No GPU is needed.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -e .                     # simulator, oracle, optimisation baselines
pip install -e ".[rl]"               # + Stable-Baselines3 and PyTorch (RL baselines)
pip install -e ".[rl,dev]"           # + pytest
```

Two environments reproduce the published results exactly (lock files):

- `.venv`, Python 3.14, `requirements.lock`: simulator, oracle, optimisation
  baselines, PPO and SAC, figures;
- `.venv-omnisafe`, Python 3.9, `requirements-omnisafe.lock`: PPO-Lagrangian
  (OmniSafe 0.5.0) and the `rliable` statistics.

Every results file records the git commit, package and solver versions, seeds
and wall-clock time in its `meta` block.

## Check the installation

```bash
pytest -q                                            # unit tests; data tests skip without the dataset
python scripts/gym_interface.py                      # API walkthrough on the bundled sample
```

## Build the benchmark dataset

Market data are not redistributed (see [DATA.md](DATA.md)). Build them once:

```bash
python scripts/auxiliary/build_greek_2025_dataset.py --year 2025
python scripts/data/build_publication_sidecars.py --year 2025
python scripts/data/fetch_open_meteo_previous_runs.py --year 2025
python scripts/data/verify_dataset.py data/processed
```

## Run the benchmark

```bash
make test                                   # all tests -> results/<subdir>/tests.json
make benchmark                              # oracles, baselines, RL, aggregates (protocol rev1)
make figures                                # figures/<subdir>/*.pdf
make benchmark PROTOCOL=configs/journal_v3.yaml   # an earlier protocol
```

[`results/COMMANDS.md`](../results/COMMANDS.md) lists the command behind every
result file.

Solvers: the optimisation baselines use CLARABEL through CVXPY and the oracle
uses HiGHS through SciPy. Both are installed with the package; no commercial
solver is needed.
