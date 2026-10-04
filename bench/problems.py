import os
import glob
import json

import numpy as np

from airdbm_core import TestAirfoils
from airdbm_core import VerifyDesigns

CONDS = {"low": (0.20, 1e6), "high": (0.40, 1e7)}
DIMS = [4, 8, 12]
BUDGET = {4: 4096, 8: 8192, 12: 12288}
SEEDS = [0, 1, 2, 3, 4]
CEILING = 350.0
MIN_W = 1e-6
COND_CODE = {0.20: "2", 0.40: "4"}            # mach -> C in ADO-<O>-<C>-<N>
SO_METHODS = ["cmaes", "de", "ga", "pso", "sobol"]
MO_METHODS = ["nsga2", "smsemoa", "moead", "optuna", "sobol"]


def make_args(mach, rey):
    return {"airfoil_db_dir": "airfoilDB", "dbm_weight_range": [0.0, 1.0],
            "dbm_normalization": "ABS_SUM", "reynolds": float(rey), "mach": float(mach),
            "clcd_ceiling": CEILING, "xfoil_evaluation": True, "xfoil_backend": "apptainer",
            "xfoil_strict": False, "xfoil_retry": 1, "parallel": True, "max_workers": 8}


def run_batch(X, args):
    X = np.clip(np.atleast_2d(np.asarray(X, float)), 0.0, 1.0)
    y = np.zeros(len(X))
    keep = np.abs(X).sum(1) >= MIN_W
    if keep.any():
        rr = TestAirfoils(X[keep], args=args, m=1)
        for j, k in enumerate(np.flatnonzero(keep)):
            o = rr[j].objectives
            c = o[0] if isinstance(o, (list, tuple)) else o
            if c is not None and np.isfinite(c) and c > 0:
                y[k] = float(c)
    return y


def evaluate(X, args):
    X = np.clip(np.atleast_2d(np.asarray(X, float)), 0.0, 1.0)
    Y = np.zeros((len(X), 2))
    ok = np.abs(X).sum(1) >= MIN_W
    if ok.any():
        res = TestAirfoils(X[ok], args=args, m=2)
        for j, i in enumerate(np.flatnonzero(ok)):
            o = res[j].objectives
            v = list(o) if isinstance(o, (list, tuple)) else [o]
            c = float(v[0]) if v[0] is not None else np.nan
            if np.isfinite(c) and c > 0:
                Y[i, 0] = c
                t = float(v[1]) if len(v) > 1 and v[1] is not None else np.nan
                Y[i, 1] = max(0.0, t) if np.isfinite(t) else 0.0
    return Y


def spec(pid):
    o, c, n = pid.split("-")[1:]
    mach, rey = next(v for v in CONDS.values() if COND_CODE[v[0]] == c)
    ndim = DIMS[int(n) - 1]
    return {"pid": pid, "m": 1 if o == "S" else 2, "ndim": ndim, "mach": mach, "rey": rey,
            "budget": BUDGET[ndim]}


# frozen gate for the runs: recheck a cand only if the frozen set dont dominate it, floor if neighbors
# dont match. improv: reads gate.json off disk on first call, ok for a one-shot script
_GATE = None
def gate_set(pid):
    global _GATE
    if _GATE is None:
        _GATE = {"so_best": {}, "mo_front": {}}
        for fn in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ADO-*", "gate.json")):
            for key, val in json.load(open(fn)).items():
                _GATE[key].update(val)
    if pid.split("-")[1] == "S":
        return "best", float(_GATE["so_best"][pid])
    return "front", _GATE["mo_front"][pid]


def screen(X, Y, pid, mach, rey, m):
    args = make_args(mach, rey)
    X = np.atleast_2d(np.asarray(X, float))
    Y = np.asarray(Y, float).reshape(len(X), -1)
    kind, ref = gate_set(pid)
    if kind == "best":
        hot = list(np.flatnonzero(Y[:, 0] > ref))
    else:
        # hot = [i for i, (a, b) in enumerate(Y) if not any(pa > a and pb > b for pa, pb in ref)]
        hot = [i for i, (a, b) in enumerate(Y)
               if a > 0 and not any(pa >= a and pb >= b and (pa > a or pb > b) for pa, pb in ref)]
    if len(hot):
        verds = VerifyDesigns(X[hot], args=args, m=m, eps=1e-6, n_dir=4, rel_tol=0.02, objective=0)
        for i, r in zip(hot, verds):
            if not r["robust"]:
                Y[i] = 0.0
    return Y if m == 2 else Y[:, 0]
