import os

import numpy as np

from airdbm_core import TestAirfoils
from airdbm_core import VerifyDesigns
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

CONDS = {"high": (0.40, 1e7)}
DIMS = [8]
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


def floor_front(X, Y, args, arch):
    # recheck a candidate the running non-dominated set does not already dominate
    hot = [i for i, (a, b) in enumerate(Y)
           if a > 0 and b >= 0 and not any(pa >= a and pb >= b for pa, pb in arch)]
    if hot:
        vs = VerifyDesigns(X[hot], args=args, m=2, eps=1e-6, n_dir=4, rel_tol=0.02, objective=0)
        for i, r in zip(hot, vs):
            if not r["robust"]:
                Y[i] = 0.0
    for a, b in Y:
        if a > 0 and b >= 0 and not any(pa >= a and pb >= b for pa, pb in arch):
            arch.append((a, b))               # archive not pruned, ok for the floor check
    return Y


def running_hv(Y):
    Y = np.atleast_2d(np.asarray(Y, float))
    out = np.empty(len(Y))
    front = []
    hv = 0.0
    for i, (a, b) in enumerate(Y):
        if a > 0 and b > 0 and not any(p >= a and q >= b for p, q in front):
            front = [(p, q) for p, q in front if not (a >= p and b >= q)]
            front.append((a, b))
            front.sort(key=lambda t: -t[0])
            hv = sum((p - (front[j + 1][0] if j + 1 < len(front) else 0.0)) * q
                     for j, (p, q) in enumerate(front))
        out[i] = hv
    return out


class MOProblem(Problem):
    def __init__(self, ndim, args, log, arch):
        super().__init__(n_var=ndim, n_obj=2, xl=np.zeros(ndim), xu=np.ones(ndim))
        self.args, self.log, self.arch = args, log, arch

    def _evaluate(self, X, out, *a, **k):
        Y = evaluate(X, self.args)
        if GATE:
            Y = floor_front(X, Y, self.args, self.arch)
        for xi, yi in zip(np.atleast_2d(X), Y):
            self.log.append((xi.copy(), float(yi[0]), float(yi[1])))
        out["F"] = -np.asarray(Y, float).reshape(-1, 2)


def run(ndim, mach, rey, seed):
    args = make_args(mach, rey)
    log, arch = [], []
    prob = MOProblem(ndim, args, log, arch)
    cx = SBX(prob=0.9, eta=20)
    mu = PM(prob=1.0 / ndim, eta=20)
    algo = NSGA2(pop_size=POP, crossover=cx, mutation=mu, eliminate_duplicates=True)
    minimize(prob, algo, ("n_evals", BUDGET[ndim]), seed=seed, verbose=False)
    log = log[:BUDGET[ndim]]
    X = np.array([r[0] for r in log])
    Y = np.array([[r[1], r[2]] for r in log])
    tag = f"MO_D{ndim}_Re{rey:.0e}_Ma{mach:g}"
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    cols = np.column_stack([X, Y, running_hv(Y)])
    head = ",".join([f"x{i}" for i in range(ndim)] + ["y_cl_cd", "y_dalpha", "running_hv"])
    np.savetxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"nsga2_seed{seed}.csv"), cols, delimiter=",", header=head, comments="")
    print(f"{tag} nsga2 seed{seed}: {len(Y)} evals, front {len(arch)}")


if __name__ == "__main__":
    for mach, rey in CONDS.values():
        for ndim in DIMS:
            for seed in SEEDS:
                run(ndim, mach, rey, seed)
