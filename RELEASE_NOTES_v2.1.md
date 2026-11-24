# PV-BESS-Gym 2.1.0 (protocol rev1)

Prepared locally; the tag `v2.1` and a Zenodo record are **not** created.

## What changed

Protocol **rev1** (`configs/journal.yaml`, results in `results/rev1/`) fixes
four modelling issues found while checking the v3 results. Each change sits
behind a flag whose library default keeps the v3 behaviour, so
`configs/journal_v3.yaml` still reproduces v3 exactly (checked: 5 baselines × 8 days ×
10 seeds, largest difference 5.7e-14 EUR).

| Flag | rev1 value | Fix |
|---|---|---|
| `market_clock` | `cet` | HEnEx and IPTO files are in CET/CEST wall clock; market values are moved one hour later to Athens time. 2025-01-01 lacks its first market hour and is dropped (362 delivery days). |
| `radiation_stamp` | `interval_centre` | Open-Meteo hourly irradiance (preceding-hour mean) is moved 30 min earlier, to the centre of its interval. |
| `id_reference` | `d1_plan` | The intraday rule and MPC expect to deliver the D-1 planned metered injection plus the PV forecast change, not the PV forecast alone. |
| `id_position_limit` | `clip` | The commercial position `x + Y` is held within the grid limits in the environment, the intraday MPC and both oracles. |

Also:
- `oracle.deliver_tasks`: ORACLE-DELIVER (`U_D`, `S_D`) is solved for T1-Nominal and T2-Forecast only.
- The oracle retries once with 4× the time limit when HiGHS returns no incumbent (column `retries`).
- The paper figures are built by `figures/src/` (`make figures`), replacing `scripts/plot_*.py`.
- New tests: `tests/test_rev1.py`.

Details and evidence: [CHANGELOG.md](CHANGELOG.md), `results/status_rev1.md`,
`results/repro_rev1.md`.

## Known issues

- Test R4 (`tests/test_realism_v3.py`) fails under rev1 on 2025-11-16, steps
  43–56 (0.002–0.010 kWh). The D-1 QP disposes of surplus PV by importing and
  exporting about 0.8 kW at the same time. This is not fixed, and the test is
  not loosened (author decision).
- ORACLE-DELIVER solves that reach the 120 s time limit are not bit-reproducible.

## Reproducing

```bash
pip install -e ".[rl]"   # see docs/INSTALL.md
make test
make benchmark      # protocol rev1
make figures
```

`results/COMMANDS.md` maps each result file to its command.
