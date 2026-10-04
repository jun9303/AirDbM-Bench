import csv
import json
import os
import numpy as np
from airdbm_core import TestAirfoils, VerifyDesigns
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.optimize import minimize

CONDS = {"low": (0.20, 1e6)}
DIMS = [8]
BUDGET = {4: 4096, 8: 8192, 12: 12288}
SEEDS = range(5)
CEILING = 350.0
MIN_W = 1e-6
GATE = False


def make_args(mach, rey):
    a = {"airfoil_db_dir": "airfoilDB", "dbm_weight_range": [0.0, 1.0],
         "dbm_normalization": "ABS_SUM", "reynolds": float(rey), "mach": float(mach),
         "clcd_ceiling": CEILING, "xfoil_evaluation": True, "xfoil_backend": "apptainer",
         "xfoil_strict": False, "xfoil_retry": 1, "parallel": True, "max_workers": 8}
    return a


def clcd(X, args):
    X = np.clip(np.atleast_2d(np.asarray(X, float)), 0.0, 1.0)
    out = np.zeros(len(X))
    ok = np.abs(X).sum(1) >= MIN_W
    idx = np.flatnonzero(ok)
    if len(idx):
        r = TestAirfoils(X[ok], args=args, m=1)
        for j, i in enumerate(idx):
            v = r[j].objectives
            v = v[0] if isinstance(v, (list, tuple)) else v
            if v is not None and np.isfinite(v) and v > 0:
                out[i] = float(v)
    return out


def screen_best(X, y, args, best):
    win = np.flatnonzero(y > best)
    if len(win) == 0:
        return y
    verds = VerifyDesigns(X[win], args=args, m=1, eps=1e-6, n_dir=4, rel_tol=0.02, objective=0)
    for i, vd in zip(win, verds):
        if not vd["robust"]:
            y[i] = 0.0
    return y


class Prob(Problem):
    def __init__(self, ndim, args, Xlog, ylog):
        super().__init__(n_var=ndim, n_obj=1, xl=np.zeros(ndim), xu=np.ones(ndim))
        self.args, self.Xlog, self.ylog, self.best = args, Xlog, ylog, 0.0

    def _evaluate(self, X, out, *a, **k):
        y = clcd(X, self.args)
        if GATE:
            y = screen_best(X, y, self.args, self.best)
        self.best = max(self.best, float(y.max()) if len(y) else 0.0)
        for xi, yi in zip(np.atleast_2d(X), y):
            self.Xlog.append(xi.copy())
            self.ylog.append(float(yi))
        out["F"] = -y.reshape(-1, 1)


def run_case(ndim, mach, rey, seed):
    args = make_args(mach, rey)
    Xlog, ylog = [], []
    prob = Prob(ndim, args, Xlog, ylog)
    # storn-price classic: pop 10*D, DE/rand/1/bin, F .8 CR .9
    algo = DE(pop_size=10 * ndim, variant="DE/rand/1/bin", F=0.8, CR=0.9)
    minimize(prob, algo, ("n_evals", BUDGET[ndim]), seed=seed, verbose=False)
    nfe = min(len(ylog), BUDGET[ndim])
    X = np.array(Xlog[:nfe])
    y = np.array(ylog[:nfe])
    rb = np.maximum.accumulate(y)
    tag = "SO_D%d_Re%.0e_Ma%g" % (ndim, rey, mach)
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "de_seed%d.csv" % seed), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x%d" % i for i in range(ndim)] + ["y_cl_cd", "running_best"])
        for a in range(nfe):
            w.writerow(list(X[a]) + [y[a], rb[a]])
    print("%s de seed%d -> %d evals, best %.2f" % (tag, seed, nfe, y.max() if nfe else 0.0))


if __name__ == "__main__":
    for mach, rey in CONDS.values():
        for ndim in DIMS:
            for seed in SEEDS:
                run_case(ndim, mach, rey, seed)
