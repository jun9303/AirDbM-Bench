import os

import numpy as np
from scipy.stats import qmc

from airdbm_core import TestAirfoils, VerifyDesigns

CONDS = {"low": (0.20, 1e6)}
DIMS = [8]
BUDGET = {4: 4096, 8: 8192, 12: 12288}
SEEDS = [0, 1, 2, 3, 4]
CEILING = 350.0
MIN_W = 1e-6
GATE = False
BATCH = 32


def make_args(mach, rey):
    return {"airfoil_db_dir": "airfoilDB", "dbm_weight_range": [0.0, 1.0],
            "dbm_normalization": "ABS_SUM", "reynolds": float(rey), "mach": float(mach),
            "clcd_ceiling": CEILING, "xfoil_evaluation": True, "xfoil_backend": "apptainer",
            "xfoil_strict": False, "xfoil_retry": 1, "parallel": True, "max_workers": 8}


def evaluate(X, args):
    X = np.clip(np.atleast_2d(np.asarray(X, float)), 0.0, 1.0)
    y = np.zeros(len(X))
    ok = np.abs(X).sum(1) >= MIN_W
    if ok.any():
        res = TestAirfoils(X[ok], args=args, m=1)
        for j, i in enumerate(np.flatnonzero(ok)):
            o = res[j].objectives
            c = o[0] if isinstance(o, (list, tuple)) else o
            if c is not None and np.isfinite(c) and c > 0:
                y[i] = float(c)
    return y


def stable(X, y, args, best):
    hot = np.flatnonzero(y > best)
    if len(hot):
        chk = VerifyDesigns(X[hot], args=args, m=1, eps=1e-6, n_dir=4, rel_tol=0.02, objective=0)
        for i, r in zip(hot, chk):
            if not r["robust"]:
                y[i] = 0.0
    return y


def run(ndim, mach, rey, seed):
    args = make_args(mach, rey)
    budget = BUDGET[ndim]
    # space-filling control, no search. scrambled sobol; budgets are 2^k so .random is balanced
    pts = qmc.Sobol(d=ndim, scramble=True, seed=seed).random(budget)
    Xlog, ylog = [], []
    best = 0.0
    done = 0
    while done < budget:
        xb = pts[done:done + BATCH]
        yb = evaluate(xb, args)
        if GATE:
            yb = stable(xb, yb, args, best)
        if len(yb):
            best = max(best, float(yb.max()))
        for xi, yi in zip(xb, yb):
            Xlog.append(xi.copy())
            ylog.append(float(yi))
        done += BATCH
    X = np.array(Xlog[:budget])
    y = np.array(ylog[:budget])
    rb = np.maximum.accumulate(y)
    tag = f"SO_D{ndim}_Re{rey:.0e}_Ma{mach:g}"
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"sobol_seed{seed}.csv"), "w") as fh:
        fh.write(",".join([f"x{i}" for i in range(ndim)] + ["y_cl_cd", "running_best"]) + "\n")
        for i in range(len(y)):
            fh.write(",".join(str(v) for v in list(X[i]) + [y[i], rb[i]]) + "\n")
    print(f"{tag} sobol {seed}: {len(y)} pts, best {best:.2f}")


if __name__ == "__main__":
    for mach, rey in CONDS.values():
        for ndim in DIMS:
            for seed in SEEDS:
                run(ndim, mach, rey, seed)
