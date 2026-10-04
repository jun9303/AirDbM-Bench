import os
import sys
import glob
import json

import numpy as np

from airdbm_core import TestAirfoils
from airdbm_core import VerifyDesigns
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.optimize import minimize

PID = "ADO-M-2-2"
MACH, REY, NDIM = 0.20, 1e6, 8
LOWER = ['ADO-M-2-1']
SO_PIDS = ['ADO-S-2-1', 'ADO-S-2-2']
CEILING = 350.0
MIN_W = 1e-6
POP = 32
NGEN = 20
N_LEVELS = 21
N_EXTEND = 4
SEEDS = [0, 1]
MAX_SWEEPS = 10
HV_TOL = 0.001
REF_OFFSET = 0.05
MO_METHODS = ["nsga2", "smsemoa", "moead", "optuna", "sobol"]
DIM_OF = {"1": 4, "2": 8, "3": 12}
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(os.getcwd(), "_work", PID)


def make_args(workers):
    return {"airfoil_db_dir": "airfoilDB", "dbm_weight_range": [0.0, 1.0],
            "dbm_normalization": "ABS_SUM", "reynolds": float(REY), "mach": float(MACH),
            "clcd_ceiling": CEILING, "xfoil_evaluation": True, "xfoil_backend": "apptainer",
            "xfoil_strict": False, "xfoil_retry": 1, "parallel": True, "max_workers": workers}


def blend(x):
    return np.round(x / max(np.abs(x).sum(), MIN_W), 12).tobytes()


def evaluate(X, args):
    X = np.clip(np.atleast_2d(np.asarray(X, float)), 0.0, 1.0)
    Y = np.zeros((len(X), 2))
    H = np.zeros(len(X), dtype=bool)
    ok = np.abs(X).sum(1) >= MIN_W
    if ok.any():
        res = TestAirfoils(X[ok], args=args, m=2)
        for j, i in enumerate(np.flatnonzero(ok)):
            H[i] = int((res[j].xfoil_result or {}).get("scan_timeouts", 0) or 0) > 0
            o = res[j].objectives
            v = list(o) if isinstance(o, (list, tuple)) else [o]
            c = float(v[0]) if v[0] is not None else np.nan
            if np.isfinite(c) and c > 0:
                Y[i, 0] = c
                t = float(v[1]) if len(v) > 1 and v[1] is not None else np.nan
                Y[i, 1] = max(0.0, t) if np.isfinite(t) else 0.0
    return Y, H


def nondom(Y):
    Y = np.atleast_2d(np.asarray(Y, float))[:, :2]
    if len(Y) == 0:
        return Y
    Y = Y[np.lexsort((-Y[:, 1], -Y[:, 0]))]
    keep = Y[:, 1] > np.maximum.accumulate(np.concatenate(([-np.inf], Y[:-1, 1])))
    return Y[keep]


def hv_2d(Y, ref=(0.0, 0.0)):
    Y = np.atleast_2d(np.asarray(Y, float))[:, :2]
    Y = Y[(Y[:, 0] > ref[0]) & (Y[:, 1] > ref[1])]
    if len(Y) == 0:
        return 0.0
    Y = Y[np.argsort(-Y[:, 0])]
    area = 0.0
    top = ref[1]
    for a, b in Y:
        if b > top:
            area += (a - ref[0]) * (b - top)
            top = b
    return float(area)


def normalize(front):
    z = np.atleast_2d(np.asarray(front, float))[:, :2].max(axis=0)
    z[z <= 0] = 1.0
    return z


def norm_hv(Y, z, ref_offset=REF_OFFSET):
    Y = np.atleast_2d(np.asarray(Y, float))[:, :2] / z
    return hv_2d(Y, ref=(-ref_offset, -ref_offset))


def pad(X):
    X = np.atleast_2d(np.asarray(X, float))
    return np.hstack([X, np.zeros((len(X), NDIM - X.shape[1]))])


def load_runs():
    Xs, Ys = [], []
    for pid in LOWER + [PID]:
        nd = DIM_OF[pid.split("-")[3]]
        for meth in MO_METHODS:
            for f in sorted(glob.glob(os.path.join(HERE, "..", pid, meth + "_seed*.csv*"))):
                a = np.atleast_2d(np.loadtxt(f, delimiter=",", skiprows=1))
                Xs.append(pad(a[:, :nd]))
                Ys.append(a[:, nd:nd + 2])
    return np.vstack(Xs), np.vstack(Ys)


def so_optima(workers):
    fn = os.path.join(WORK, "so_optima.json")
    if os.path.exists(fn):
        d = json.load(open(fn))
        return np.array(d["X"]).reshape(-1, NDIM), np.array(d["Y"]).reshape(-1, 2)
    X = np.vstack([pad(json.load(open(os.path.join(HERE, "..", p, "data.json")))["x_ref"]) for p in SO_PIDS])
    v = VerifyDesigns(X, args=make_args(workers), m=2, eps=1e-6, n_dir=4, rel_tol=0.02, objective=0)
    keep = [i for i, r in enumerate(v) if r["robust"]]
    Y = np.array([[max(0.0, float(v[i]["objectives"][0])), max(0.0, float(v[i]["objectives"][1]))] for i in keep])
    os.makedirs(WORK, exist_ok=True)
    json.dump({"X": X[keep].tolist(), "Y": Y.tolist()}, open(fn, "w"))
    return X[keep].reshape(-1, NDIM), Y.reshape(-1, 2)


def gate_front():
    return np.asarray(json.load(open(os.path.join(HERE, "gate.json")))["mo_front"][PID], float).reshape(-1, 2)


def gate_refine():
    out = set()
    for pid in LOWER + [PID]:
        nd = DIM_OF[pid.split("-")[3]]
        files = [f for m in MO_METHODS for f in sorted(glob.glob(os.path.join(HERE, "..", pid, m + "_seed*.csv*")))]
        a = np.vstack([np.atleast_2d(np.loadtxt(f, delimiter=",", skiprows=1)) for f in files])
        X, Y = pad(a[:, :nd]), a[:, nd:nd + 2]
        G = np.asarray(json.load(open(os.path.join(HERE, "..", pid, "gate.json")))["mo_front"][pid], float).reshape(-1, 2)
        bad = {(float(p), float(q)) for p, q in nondom(Y) if not ((G[:, 0] >= p) & (G[:, 1] >= q)).any()}
        out |= {x.tobytes().hex() for x, y in zip(X, Y) if (float(y[0]), float(y[1])) in bad}
    return out


def levels(front):
    lo, hi = float(front[:, 1].min()), float(front[:, 1].max())
    step = (hi - lo) / (N_LEVELS - 1)
    return sorted(set(round(lo + i * step, 1) for i in range(N_LEVELS + N_EXTEND)))


def csv_path(seed):
    return os.path.join(HERE, f"refinement_seed{seed}.csv")


def read_samples(seed):
    p = csv_path(seed)
    if not os.path.exists(p):
        p = p + ".gz"
    if not os.path.exists(p):
        return np.zeros((0, NDIM)), np.zeros((0, 2)), np.zeros(0), np.zeros(0)
    a = np.atleast_2d(np.loadtxt(p, delimiter=",", skiprows=1))
    return a[:, :NDIM], a[:, NDIM:NDIM + 2], a[:, NDIM + 2], a[:, NDIM + 3]


def append_rows(seed, X, Y, level, sweep):
    p = csv_path(seed)
    new = not os.path.exists(p)
    with open(p, "a") as fh:
        if new:
            fh.write(",".join([f"x{i}" for i in range(NDIM)] + ["y_cl_cd", "y_dalpha", "level", "sweep"]) + "\n")
        for x, y in zip(X, Y):
            fh.write(",".join(repr(float(v)) for v in list(x) + list(y) + [level, sweep]) + "\n")


def load_state():
    fn = os.path.join(WORK, "state.json")
    if os.path.exists(fn):
        return json.load(open(fn))
    return {"verified": {}, "fronts": {"0": gate_front().tolist()}, "grids": {"1": levels(gate_front())},
            "stop": False, "log": []}


def save_state(state):
    os.makedirs(WORK, exist_ok=True)
    json.dump(state, open(os.path.join(WORK, "state.json"), "w"))


def warm_start(X, Y, level, skip=frozenset()):
    viol = np.maximum(0.0, level - Y[:, 1])
    order = np.lexsort((-Y[:, 0], viol))
    seen, keep = set(), []
    for i in order:
        k = X[i].tobytes()
        if k in seen or k.hex() in skip:
            continue
        seen.add(k)
        keep.append(i)
        if len(keep) == POP:
            break
    return X[keep]


class LevelProblem(Problem):
    def __init__(self, level, args, cache, log, hung):
        super().__init__(n_var=NDIM, n_obj=1, n_ieq_constr=1, xl=np.zeros(NDIM), xu=np.ones(NDIM))
        self.level, self.args, self.cache, self.log, self.hung = level, args, cache, log, hung

    def _evaluate(self, X, out, *a, **k):
        X = np.atleast_2d(X)
        Y = np.zeros((len(X), 2))
        new = []
        for i, x in enumerate(X):
            y = self.cache.get(x.tobytes())
            if y is None:
                new.append(i)
            else:
                Y[i] = y
        if new:
            run = [i for i in new if blend(X[i]) not in self.hung]
            Yr, Hr = evaluate(X[run], self.args)
            for j, i in enumerate(run):
                if Hr[j]:
                    self.hung[blend(X[i])] = Yr[j]
            Yn = [Yr[run.index(i)] if i in run else self.hung[blend(X[i])] for i in new]
            for j, i in enumerate(new):
                Y[i] = Yn[j]
                self.cache[X[i].tobytes()] = Yn[j]
                self.log.append((X[i].copy(), Yn[j].copy()))
        out["F"] = -Y[:, :1]
        out["G"] = (self.level - Y[:, 1]).reshape(-1, 1)


def run_sweep(seed, sweep, workers=POP):
    state = load_state()
    if state["stop"]:
        return
    grid = state["grids"][str(sweep)]
    failed = {h for h, v in state["verified"].items() if not v} | gate_refine()
    args = make_args(workers)
    Xr, Yr = load_runs()
    Xo, Yo = so_optima(workers)
    Xs, Ys, _, _ = read_samples(seed)
    X, Y = np.vstack([Xr, Xo, Xs]), np.vstack([Yr, Yo, Ys])
    cache = {x.tobytes(): y for x, y in zip(X, Y)}
    hung = {}
    pfile = os.path.join(WORK, f"progress_seed{seed}.json")
    prog = json.load(open(pfile)) if os.path.exists(pfile) else {}
    done = set(prog.get(str(sweep), []))
    for j, level in enumerate(grid):
        if level in done:
            continue
        log = []
        prob = LevelProblem(level, args, cache, log, hung)
        algo = GA(pop_size=POP, sampling=warm_start(X, Y, level, failed), crossover=SBX(prob=0.9, eta=20),
                  mutation=PM(prob=1.0 / NDIM, eta=20), eliminate_duplicates=True)
        ga_seed = 2 * (1000 * sweep + j) + seed
        minimize(prob, algo, ("n_gen", NGEN), seed=ga_seed, verbose=False)
        if log:
            Xn = np.array([r[0] for r in log])
            Yn = np.array([r[1] for r in log])
            append_rows(seed, Xn, Yn, level, sweep)
            X, Y = np.vstack([X, Xn]), np.vstack([Y, Yn])
        done.add(level)
        prog[str(sweep)] = sorted(done)
        os.makedirs(WORK, exist_ok=True)
        json.dump(prog, open(pfile, "w"))
        print(f"{PID} seed{seed} sweep{sweep} level {level}: {len(log)} new", flush=True)


def check(sweep, workers=128):
    state = load_state()
    if state["stop"]:
        return
    args = make_args(workers)
    G = gate_front()
    parts = [read_samples(s) for s in SEEDS]
    Xs = np.vstack([p[0][p[3] <= sweep] for p in parts])
    Ys = np.vstack([p[1][p[3] <= sweep] for p in parts])
    gset = {(float(a), float(b)) for a, b in G}
    ver = state["verified"]
    n_checked = 0
    while True:
        ok, nfail = [], {}
        for i in range(len(Xs)):
            key = (float(Ys[i, 0]), float(Ys[i, 1]))
            if ver.get(Xs[i].tobytes().hex(), True):
                ok.append(i)
            else:
                nfail[key] = nfail.get(key, 0) + 1
        group = {}
        for i in ok:
            group.setdefault((float(Ys[i, 0]), float(Ys[i, 1])), []).append(i)
        F = nondom(np.vstack([G, Ys[ok]]))
        cand, seen = [], set()
        for a, b in F:
            key = (float(a), float(b))
            if key in gset:
                continue
            g = group[key]
            if Xs[g[0]].tobytes().hex() in ver:
                continue
            take, k = min(POP, max(1, nfail.get(key, 0))), 0
            for i in g:
                h = Xs[i].tobytes().hex()
                if h in ver or h in seen:
                    continue
                seen.add(h)
                cand.append(i)
                k += 1
                if k == take:
                    break
        if not cand:
            break
        v = VerifyDesigns(Xs[cand], args=args, m=2, eps=1e-6, n_dir=4, rel_tol=0.02, objective=0)
        for i, r in zip(cand, v):
            ver[Xs[i].tobytes().hex()] = bool(r["robust"])
        n_checked += len(cand)
    prev = np.asarray(state["fronts"][str(sweep - 1)], float)
    z = normalize(F)
    hv_now, hv_prev = norm_hv(F, z), norm_hv(prev, z)
    growth = (hv_now - hv_prev) / hv_prev
    dn = abs(len(F) - len(prev)) / len(prev)
    state["fronts"][str(sweep)] = F.tolist()
    state["grids"][str(sweep + 1)] = levels(F)
    state["stop"] = bool(growth < HV_TOL or sweep >= MAX_SWEEPS)
    state["log"].append({"sweep": sweep, "n_front": len(F), "hv": hv_now, "hv_prev": hv_prev, "growth": growth,
                         "dn": dn, "n_checked": n_checked,
                         "n_failed": sum(1 for val in ver.values() if not val), "stop": state["stop"]})
    save_state(state)
    print(json.dumps(state["log"][-1]), flush=True)


def finalize(workers=128):
    state = load_state()
    parts = [read_samples(s) for s in SEEDS]
    Xs = np.vstack([p[0] for p in parts])
    Ys = np.vstack([p[1] for p in parts])
    ok = np.array([state["verified"].get(x.tobytes().hex(), False) for x in Xs], dtype=bool)
    Xr, Yr = load_runs()
    Xo, Yo = so_optima(workers)
    gset = {(float(a), float(b)) for a, b in gate_front()}
    gi = [i for i, y in enumerate(Yr) if (float(y[0]), float(y[1])) in gset]
    Xl, Yl = np.zeros((0, NDIM)), np.zeros((0, 2))
    if LOWER:
        low = json.load(open(os.path.join(HERE, "..", LOWER[-1], "data.json")))
        Xl, Yl = pad(low["front_X"]), np.asarray(low["front_Y"], float).reshape(-1, 2)
    X = np.vstack([Xs[ok], Xo, Xr[gi], Xl])
    Y = np.vstack([Ys[ok], Yo, Yr[gi], Yl])
    first = {}
    for i, y in enumerate(Y):
        first.setdefault((float(y[0]), float(y[1])), i)
    F = nondom(Y)
    idx = [first[(float(a), float(b))] for a, b in F]
    rec = {"D": NDIM, "mach": MACH, "reynolds": REY, "budget": {4: 4096, 8: 8192, 12: 12288}[NDIM],
           "n_front": len(idx), "front_Y": Y[idx].tolist(), "front_X": X[idx].tolist(),
           "so_Y": Yo.tolist(), "so_X": Xo.tolist(),
           "failed_X": [np.frombuffer(bytes.fromhex(h)).tolist() for h, v in state["verified"].items() if not v]}
    with open(os.path.join(HERE, "data.json"), "w") as fh:
        json.dump(rec, fh, indent=2)
        fh.write("\n")
    print(f"{PID}: final front {len(idx)} points", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        run_sweep(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "check":
        check(int(sys.argv[2]))
    elif len(sys.argv) > 1 and sys.argv[1] == "finalize":
        finalize()
    else:
        for sweep in range(1, MAX_SWEEPS + 1):
            if load_state()["stop"]:
                break
            for seed in SEEDS:
                run_sweep(seed, sweep)
            check(sweep)
        finalize()
