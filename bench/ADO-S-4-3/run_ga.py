import os

import numpy as np

from airdbm_core import TestAirfoils
from airdbm_core import VerifyDesigns
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

CONDS = {"high": (0.40, 1e7)}
DIMS = [12]
BUDGET = {4: 4096, 8: 8192, 12: 12288}
SEEDS = [0, 1, 2, 3, 4]
CEILING = 350.0
MIN_W = 1e-6
GATE = False
POP = 100


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


def drop_unstable(X, y, args, best):
    for i in np.flatnonzero(y > best):        # record-breakers, one at a time
        v = VerifyDesigns(X[i:i + 1], args=args, m=1, eps=1e-6, n_dir=4, rel_tol=0.02, objective=0)
        if not v[0]["robust"]:
            y[i] = 0.0
    return y


class GAProblem(Problem):
    def __init__(self, ndim, args, log):
        super().__init__(n_var=ndim, n_obj=1, xl=np.zeros(ndim), xu=np.ones(ndim))
        self.args = args
        self.log = log
        self.best = 0.0

    def _evaluate(self, X, out, *a, **k):
        y = run_batch(X, self.args)
        if GATE:
            y = drop_unstable(X, y, self.args, self.best)
        if len(y):
            self.best = max(self.best, float(y.max()))
        for xi, yi in zip(np.atleast_2d(X), y):
            self.log.append((xi.copy(), float(yi)))
        out["F"] = -y.reshape(-1, 1)


def run(ndim, mach, rey, seed):
    args = make_args(mach, rey)
    log = []
    prob = GAProblem(ndim, args, log)
    # deb 2002: SBX pc .9 eta 20, PM pm 1/n eta 20
    cx = SBX(prob=0.9, eta=20)
    mu = PM(prob=1.0 / ndim, eta=20)
    algo = GA(pop_size=POP, crossover=cx, mutation=mu, eliminate_duplicates=True)
    minimize(prob, algo, ("n_evals", BUDGET[ndim]), seed=seed, verbose=False)
    log = log[:BUDGET[ndim]]
    X = np.array([r[0] for r in log])
    y = np.array([r[1] for r in log])
    tag = f"SO_D{ndim}_Re{rey:.0e}_Ma{mach:g}"
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    cols = np.column_stack([X, y, np.maximum.accumulate(y)])
    head = ",".join([f"x{i}" for i in range(ndim)] + ["y_cl_cd", "running_best"])
    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"ga_seed{seed}.csv"), cols, delimiter=",", header=head, comments="")
    print(f"{tag} ga seed{seed}: best Cl/Cd {y.max() if len(y) else 0.0:.2f}")


if __name__ == "__main__":
    for mach, rey in CONDS.values():
        for ndim in DIMS:
            for seed in SEEDS:
                run(ndim, mach, rey, seed)
