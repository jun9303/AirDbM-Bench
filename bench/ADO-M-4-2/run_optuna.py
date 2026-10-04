import csv
import os

import numpy as np
import optuna

from airdbm_core import TestAirfoils, VerifyDesigns

CONDS = {"high": (0.40, 1e7)}
DIMS = [8]
BUDGET = {4: 4096, 8: 8192, 12: 12288}
SEEDS = [0, 1, 2, 3, 4]
CEILING = 350.0
MIN_W = 1e-6
GATE = False
BATCH = 8


def make_args(mach, rey):
    return {"airfoil_db_dir": "airfoilDB", "dbm_weight_range": [0.0, 1.0],
            "dbm_normalization": "ABS_SUM", "reynolds": float(rey), "mach": float(mach),
            "clcd_ceiling": CEILING, "xfoil_evaluation": True, "xfoil_backend": "apptainer",
            "xfoil_strict": False, "xfoil_retry": 1, "parallel": True, "max_workers": 8}


def two_obj(X, args):
    X = np.clip(np.atleast_2d(np.asarray(X, float)), 0.0, 1.0)
    Y = np.zeros((len(X), 2))
    ok = np.abs(X).sum(1) >= MIN_W
    if ok.any():
        rr = TestAirfoils(X[ok], args=args, m=2)
        for j, i in enumerate(np.flatnonzero(ok)):
            o = rr[j].objectives
            v = list(o) if isinstance(o, (list, tuple)) else [o]
            c = float(v[0]) if v[0] is not None else np.nan
            if np.isfinite(c) and c > 0:
                Y[i, 0] = c
                t = float(v[1]) if len(v) > 1 and v[1] is not None else np.nan
                Y[i, 1] = max(0.0, t) if np.isfinite(t) else 0.0
    return Y


def screen(X, Y, args, front):
    hot = [i for i, (a, b) in enumerate(Y)
           if a > 0 and b >= 0 and not any(pa >= a and pb >= b for pa, pb in front)]
    if hot:
        vs = VerifyDesigns(X[hot], args=args, m=2, eps=1e-6, n_dir=4, rel_tol=0.02, objective=0)
        for i, r in zip(hot, vs):
            if not r["robust"]:
                Y[i] = 0.0
    for a, b in Y:
        if a > 0 and b >= 0 and not any(pa >= a and pb >= b for pa, pb in front):
            front.append((a, b))
    return Y


def run_hv(Y):
    Y = np.atleast_2d(np.asarray(Y, float))
    out = np.empty(len(Y))
    nd, hv = [], 0.0
    for i, (a, b) in enumerate(Y):
        if a > 0 and b > 0 and not any(p >= a and q >= b for p, q in nd):
            nd = [(p, q) for p, q in nd if not (a >= p and b >= q)]
            nd.append((a, b))
            nd.sort(key=lambda t: -t[0])
            hv = sum((p - (nd[j + 1][0] if j + 1 < len(nd) else 0.0)) * q
                     for j, (p, q) in enumerate(nd))
        out[i] = hv
    return out


def run(ndim, mach, rey, seed):
    args = make_args(mach, rey)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    # tpe ask/tell in batches of 8, maximize both objectives
    study = optuna.create_study(directions=["maximize", "maximize"],
                                sampler=optuna.samplers.TPESampler(multivariate=True, seed=seed))
    hist, front = [], []
    budget = BUDGET[ndim]
    done = 0
    while done < budget:
        n = min(BATCH, budget - done)
        trials = [study.ask() for _ in range(n)]
        xb = np.array([[t.suggest_float(f"x{i}", 0.0, 1.0) for i in range(ndim)] for t in trials])
        yb = two_obj(xb, args)
        if GATE:
            yb = screen(xb, yb, args, front)
        for t, yr in zip(trials, yb):
            study.tell(t, [float(yr[0]), float(yr[1])])
        for xi, yi in zip(xb, yb):
            hist.append((xi.copy(), float(yi[0]), float(yi[1])))
        done += n
    X = np.array([r[0] for r in hist])
    Y = np.array([[r[1], r[2]] for r in hist])
    hv = run_hv(Y)
    tag = f"MO_D{ndim}_Re{rey:.0e}_Ma{mach:g}"
    os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"optuna_seed{seed}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"x{i}" for i in range(ndim)] + ["y_cl_cd", "y_dalpha", "running_hv"])
        for r in range(len(Y)):
            w.writerow(list(X[r]) + [Y[r, 0], Y[r, 1], hv[r]])
    print(f"{tag} optuna seed{seed}: {len(Y)} trials")


if __name__ == "__main__":
    for mach, rey in CONDS.values():
        for ndim in DIMS:
            for seed in SEEDS:
                run(ndim, mach, rey, seed)
