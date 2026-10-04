import os

import numpy as np

from airdbm_core import TestAirfoils, VerifyDesigns
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.pso import PSO
from pymoo.optimize import minimize

CONDS = {"low": (0.20, 1e6)}
DIMS = [8]
BUDGET = {4: 4096, 8: 8192, 12: 12288}
SEEDS = [0, 1, 2, 3, 4]
CEILING = 350.0
MIN_W = 1e-6
GATE = False


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


class P(Problem):
    def __init__(self, ndim, args, log):
        super().__init__(n_var=ndim, n_obj=1, xl=np.zeros(ndim), xu=np.ones(ndim))
        self.args, self.log, self.best = args, log, 0.0

    def _evaluate(self, X, out, *a, **k):
        y = evaluate(X, self.args)
        if GATE:
            y = stable(X, y, self.args, self.best)
        if len(y):
            self.best = max(self.best, float(y.max()))
        for xi, yi in zip(np.atleast_2d(X), y):
            self.log.append((xi.copy(), float(yi)))
        out["F"] = -y.reshape(-1, 1)


def run(ndim, mach, rey, seed):
    args = make_args(mach, rey)
    log = []
    prob = P(ndim, args, log)
    algo = PSO()          # pymoo defaults: swarm 25, w .9, c1=c2 2.0
    minimize(prob, algo, ("n_evals", BUDGET[ndim]), seed=seed, verbose=False)
    log = log[:BUDGET[ndim]]
    X = np.array([r[0] for r in log])
    y = np.array([r[1] for r in log])
    tag = f"SO_D{ndim}_Re{rey:.0e}_Ma{mach:g}"
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    cols = np.column_stack([X, y, np.maximum.accumulate(y)])
    head = ",".join([f"x{i}" for i in range(ndim)] + ["y_cl_cd", "running_best"])
    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"pso_seed{seed}.csv"), cols, delimiter=",", header=head, comments="")
    print(tag, "pso", seed, len(y))


if __name__ == "__main__":
    for mach, rey in CONDS.values():
        for ndim in DIMS:
            for seed in SEEDS:
                run(ndim, mach, rey, seed)
