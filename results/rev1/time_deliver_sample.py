"""Time ORACLE-DELIVER under rev1 (position bound) on 13 T1 test days, seed 0 -> compute estimate."""
import os, sys, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "1")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pandas as pd
from joblib import Parallel, delayed
from pv_bess_gym.benchmark.tasks import load_protocol, make_task_env, split_days

def job(day):
    warnings.filterwarnings("ignore")
    from pv_bess_gym.oracle import solve_oracle
    p = load_protocol(); env = make_task_env("T1-Nominal")
    r = solve_oracle(env, day, soc_init=0.5, activation_seed=0, allow_imbalance=False, deliver_epsilon_kwh=0.0,
                     time_limit_s=float(p["oracle"]["time_limit_s"]), mip_rel_gap=float(p["oracle"]["mip_rel_gap"]))
    return {"day": day, "U_D": r.upper_bound_eur, "gap": r.mip_gap, "seconds": r.solve_seconds, "retries": r.retries}

days = split_days(make_task_env("T1-Nominal"), "test", load_protocol())[::7][:13]
df = pd.DataFrame(Parallel(n_jobs=1)(delayed(job)(d) for d in days))
df.to_csv(Path(__file__).with_name("time_deliver_sample.csv"), index=False)
print(df.to_string()); print("mean s %.1f median %.1f max %.1f" % (df.seconds.mean(), df.seconds.median(), df.seconds.max()))
