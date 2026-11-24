#!/bin/zsh
# rev1 full run (author decision 2026-11-18): ORACLE + baselines all tasks, RL retrained,
# ORACLE-DELIVER T1/T2 test. Each stage logs to results/rev1/logs/<stage>.log; exit status and
# wall time go to results/rev1/logs/stages.tsv (input for repro_rev1.md).
cd "$(dirname "$0")/../.."
export PVBESS_PROTOCOL=configs/journal.yaml
L=results/rev1/logs; mkdir -p $L
TASKS=(T1-Nominal T2-Forecast T3-Gate T4-Ageing T5-Combined)
while pgrep -f "run_journal.py oracle " >/dev/null; do sleep 60; done
stage() { local name=$1; shift; local t0=$(date +%s); "$@" > $L/$name.log 2>&1; local rc=$?
  echo -e "$name\t$rc\t$(( $(date +%s) - t0 ))\t$*" >> $L/stages.tsv; return $rc; }
stage oracle_check  .venv/bin/python scripts/run_journal.py oracle --tasks $TASKS --jobs 12 || exit 1
stage baselines     .venv/bin/python scripts/run_journal.py baselines --tasks $TASKS --jobs 12 || exit 1
stage rl_tune       .venv/bin/python scripts/rl/run_all.py tune --jobs 12 --python .venv/bin/python --python-omnisafe .venv-omnisafe/bin/python || exit 1
stage rl_main       .venv/bin/python scripts/rl/run_all.py main --tasks $TASKS --jobs 12 --python .venv/bin/python --python-omnisafe .venv-omnisafe/bin/python || exit 1
stage rl_collect    .venv/bin/python scripts/rl/collect_rl.py || exit 1
stage oracle_deliver .venv/bin/python scripts/run_journal.py oracle_deliver --tasks T1-Nominal T2-Forecast --jobs 12 || exit 1
stage aggregate     .venv/bin/python scripts/run_journal.py aggregate --tasks $TASKS || exit 1
stage summary       .venv-omnisafe/bin/python scripts/summarise_journal.py --tasks $TASKS || exit 1
stage compute       .venv/bin/python scripts/measure_compute.py || exit 1
echo done > $L/ALL_DONE
