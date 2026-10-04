import time
import os

import numpy as np

from airdbm_core import TestAirfoils, VerifyDesigns
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.optimize import minimize

CONDS = {"low": (0.20, 1e6)}
DIMS = [4]
BUDGET = {4: 4096, 8: 8192, 12: 12288}
SEEDS = [0, 1, 2, 3, 4]
CEILING = 350.0
MIN_W = 1e-6
GATE = False


def base_args(mach, re):
    return {"airfoil_db_dir": "airfoilDB", "dbm_weight_range": [0.0, 1.0],
            "dbm_normalization": "ABS_SUM", "reynolds": float(re), "mach": float(mach),
            "clcd_ceiling": CEILING, "xfoil_evaluation": True, "xfoil_backend": "apptainer",
            "xfoil_strict": False, "xfoil_retry": 1, "parallel": True, "max_workers": 8}


def evaluate(X, args):
    X = np.clip(np.atleast_2d(np.asarray(X, float)), 0.0, 1.0)
    y = np.zeros(len(X))
    good = np.abs(X).sum(1) >= MIN_W          # skip degenerate weight sums
    if good.any():
        res = TestAirfoils(X[good], args=args, m=1)
        for j, k in enumerate(np.flatnonzero(good)):
            o = res[j].objectives
            c = o[0] if isinstance(o, (list, tuple)) else o
            if c is not None and np.isfinite(c) and c > 0:
                y[k] = float(c)
    return y


def floor_flukes(X, y, args, best):
    hot = np.flatnonzero(y > best)            # recheck only if it beats best
    if len(hot):
        chk = VerifyDesigns(X[hot], args=args, m=1, eps=1e-6, n_dir=4, rel_tol=0.02, objective=0)
        for i, r in zip(hot, chk):
            if not r["robust"]:
                y[i] = 0.0
    return y


class DbMProblem(Problem):
    def __init__(self, ndim, args, hist):
        super().__init__(n_var=ndim, n_obj=1, xl=np.zeros(ndim), xu=np.ones(ndim))
        self.args = args
        self.hist = hist
        self.best = 0.0

    def _evaluate(self, X, out, *a, **k):
        y = evaluate(X, self.args)
        if GATE:
            y = floor_flukes(X, y, self.args, self.best)
        if len(y):
            self.best = max(self.best, float(y.max()))
        for xi, yi in zip(np.atleast_2d(X), y):
            self.hist.append((xi.copy(), float(yi)))
        out["F"] = -y.reshape(-1, 1)          # pymoo minimizes, we maximize Cl/Cd


def run(ndim, mach, re, seed):
    t0 = time.time()
    args = base_args(mach, re)
    hist = []
    prob = DbMProblem(ndim, args, hist)
    # hansen sigma 0.3 for the [0,1] box, IPOP restarts for the multimodal D=12
    algo = CMAES(sigma=0.3, restarts=2, incpopsize=2)
    res = minimize(prob, algo, ("n_evals", BUDGET[ndim]), seed=seed, verbose=False)
    hist = hist[:BUDGET[ndim]]
    X = np.array([h[0] for h in hist])
    y = np.array([h[1] for h in hist])
    tag = f"SO_D{ndim}_Re{re:.0e}_Ma{mach:g}"
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    cols = np.column_stack([X, y, np.maximum.accumulate(y)])
    head = ",".join([f"x{i}" for i in range(ndim)] + ["y_cl_cd", "running_best"])
    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"cmaes_seed{seed}.csv"), cols, delimiter=",", header=head, comments="")
    print(f"{tag} cmaes/{seed}: {len(y)} evals, best {prob.best:.2f}")


if __name__ == "__main__":
    for mach, re in CONDS.values():
        for ndim in DIMS:
            for seed in SEEDS:
                run(ndim, mach, re, seed)
