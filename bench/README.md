# AirDbM-Bench problems

Every problem calls `TestAirfoils` with one frozen set of arguments. For a problem at Mach number `mach` and chord Reynolds number `rey`, the run scripts and `problems.py` build them with `make_args`:

~~~python
args = {"airfoil_db_dir": "airfoilDB", "dbm_weight_range": [0.0, 1.0],
        "dbm_normalization": "ABS_SUM", "reynolds": float(rey), "mach": float(mach),
        "clcd_ceiling": 350.0, "xfoil_evaluation": True, "xfoil_backend": "apptainer",
        "xfoil_strict": False, "xfoil_retry": 1, "parallel": True, "max_workers": 8}
~~~

The remaining arguments keep their defaults, so a problem of dimension `D` uses the first `D` of the twelve default baselines, `n_crit` = 9, a scan from 0 to 45 degrees, 160 panel nodes, 200 iterations per angle and a 60-second timeout. `max_workers` only sets how many XFOIL processes run at once and does not change any value.

Single-objective problems call `TestAirfoils(X, args=args, m=1)` and return peak `Cl/Cd`; multi-objective problems use `m=2` and add the stall margin. The budget is `1024·D` evaluations.

| problem | objectives | D | Ma | Re_c | budget | setting |
|---|---|---|---|---|---|---|
| ADO-S-2-1 | Cl/Cd | 4 | 0.20 | 1e6 | 4096 | wind-tunnel model |
| ADO-S-2-2 | Cl/Cd | 8 | 0.20 | 1e6 | 8192 | wind-tunnel model |
| ADO-S-2-3 | Cl/Cd | 12 | 0.20 | 1e6 | 12288 | wind-tunnel model |
| ADO-S-4-1 | Cl/Cd | 4 | 0.40 | 1e7 | 4096 | regional turboprop, cruise |
| ADO-S-4-2 | Cl/Cd | 8 | 0.40 | 1e7 | 8192 | regional turboprop, cruise |
| ADO-S-4-3 | Cl/Cd | 12 | 0.40 | 1e7 | 12288 | regional turboprop, cruise |
| ADO-M-2-1 | Cl/Cd, Δα | 4 | 0.20 | 1e6 | 4096 | wind-tunnel model |
| ADO-M-2-2 | Cl/Cd, Δα | 8 | 0.20 | 1e6 | 8192 | wind-tunnel model |
| ADO-M-2-3 | Cl/Cd, Δα | 12 | 0.20 | 1e6 | 12288 | wind-tunnel model |
| ADO-M-4-1 | Cl/Cd, Δα | 4 | 0.40 | 1e7 | 4096 | regional turboprop, cruise |
| ADO-M-4-2 | Cl/Cd, Δα | 8 | 0.40 | 1e7 | 8192 | regional turboprop, cruise |
| ADO-M-4-3 | Cl/Cd, Δα | 12 | 0.40 | 1e7 | 12288 | regional turboprop, cruise |

The call is wrapped by `run_batch` (one objective) or `evaluate` (two objectives) in `problems.py`. The wrapper clips `x` to `[0, 1]` and gives the objective floor, 0 or (0, 0), to:

- a design with weights that add up to less than `1e-6`, which is not sent to XFOIL;
- a design with a returned `Cl/Cd` that is missing, not finite or not positive.

A negative or missing stall margin of a valid design is set to 0. `spec(pid)` decodes a problem ID into `m`, `ndim`, `mach`, `rey` and `budget`.

`screen(X, Y, pid, mach, rey, m)` is the stability gate. It reads `gate.json` of each problem folder.

`screen` re-checks a candidate with `VerifyDesigns` (`eps` 1e-6, the `2D` axis neighbors, 2 % tolerance on `Cl/Cd`) only when the gate does not already account for it, that is, when its `Cl/Cd` exceeds the gated SO best or no gated MO point dominates it. A candidate that fails the check gets the objective floor.

The released runs used raw values during the search (`GATE = False` in the run scripts), and the gate files were built from the pooled runs after the campaign with the same check.

Each `ADO-*` folder contains the following.

- The five run scripts `run_<method>.py` are set to this problem's Ma, Re and D. Run them from the repository root after `pip install -e ".[bench]"`, e.g. `python bench/ADO-M-2-1/run_nsga2.py`. Each writes `<method>_seed<k>.csv` for seeds 0 to 4 into the same folder.
- The 25 histories `<method>_seed<k>.csv.gz` store one row per evaluation in evaluation order. The columns are `x0 … x{D-1}, y_cl_cd, running_best` for SO and `x0 … x{D-1}, y_cl_cd, y_dalpha, running_hv` for MO, where `running_hv` is the hypervolume above the origin in raw objective units.
- `gate.json` stores the verified best `Cl/Cd` (SO) or the verified non-dominated front (MO) of the released runs before refinement. An MO front also includes the SO optima of the same condition that pass the check at `m=2`.
- `data.json` stores the reference target: `y_ref` and `x_ref` for SO, with `y_ref_own` the best found by that dimension's own runs, and `front_Y` and `front_X` for MO.
- The MO folders also contain `refine.py` and its two sample files `refinement_seed0.csv.gz` and `refinement_seed1.csv.gz`.

The GitHub repository leaves these histories and refinement samples out; they are distributed in the data archive `data.tar.gz`, which unpacks into `bench/` from the repository root.

The run scripts use these optimizer settings and leave the rest at the library defaults (pymoo 0.6.2, cma 4.4.4, Optuna 4.9.0, SciPy 1.15.3):

| method | settings |
|---|---|
| CMA-ES (SO) | `sigma` 0.3, 2 restarts with the population doubled each time, default population `4 + floor(3 ln D)` |
| DE (SO) | population `10D`, `DE/rand/1/bin`, `F` 0.8, `CR` 0.9 |
| GA (SO) | population 100, SBX (0.9, `eta` 20), polynomial mutation (1/D, `eta` 20), duplicate elimination |
| PSO (SO) | pymoo defaults: swarm 25, `w` 0.9, `c1` = `c2` = 2.0 |
| NSGA-II (MO) | population 100, the GA operators, duplicate elimination |
| SMS-EMOA (MO) | population 100, one offspring per step, the GA operators |
| MOEA/D (MO) | 100 uniform reference directions, 20 neighbors, neighbor mating probability 0.9, the GA operators |
| TPE (MO) | Optuna `TPESampler(multivariate=True)`, batches of 8 |
| Sobol (SO, MO) | scrambled Sobol from `scipy.stats.qmc`, batches of 32 |

`refine.py` builds the MO reference front from the released runs. It splits the stall-margin range of the current verified front into 21 levels, rounded to 0.1 degree, and adds 4 levels above it, 25 per sweep.

At each level, in ascending order, a genetic algorithm (population 32, 20 generations, the SBX and polynomial mutation settings of the released GA) maximizes `Cl/Cd` subject to `Δα` at or above the level. It starts from the 32 best known designs for that level, leaving out any design that has failed the stability check at the gate or in an earlier sweep.

Two chains run independently, and a design already known to a chain is not evaluated again. When XFOIL reaches its timeout limit on a design due to a failure in flow solution convergence, no new XFOIL run is performed for later candidates with the same weights from the GA.

Chain `k` (0 or 1) uses seed `2·(1000·sweep + j) + k` at level `j` of a sweep, because a level that starts from the same designs as an earlier one would otherwise repeat that run. Every evaluation uses the same `TestAirfoils` arguments as the run scripts.

After each sweep over the levels, every new point on the pooled front is checked with `VerifyDesigns`, a point that fails is removed, and the levels are set again from the new front. Sweeps stop once one sweep raises the normalized hypervolume by less than 0.1 %, or after 10 sweeps.

The sample files list every evaluation of a chain with the columns `x0 … x{D-1}, y_cl_cd, y_dalpha, level, sweep`.

The final front in `data.json` is the non-dominated set of the verified refinement points, the gate front, the verified SO optima and the final front of the next lower dimension. `data.json` also lists the verified SO optima on both objectives (`so_X`, `so_Y`) and the refinement designs that failed the stability check (`failed_X`).

Run `refine.py` from the repository root with `python bench/ADO-M-2-1/refine.py`, or step by step with `sweep <seed> <k>`, `check <k>` and `finalize`.

`cost.json` lists the core-hours of each released optimizer run and their total, 15,704.5 core-hours; it does not include the refinement.

Two details of the histories affect a reproduction:

- `cma` treats seed 0 as a request for a clock-based seed, so the `cmaes_seed0` histories cannot be repeated, while seeds 1 to 4 reproduce.
- `ADO-S-4-1/cmaes_seed3.csv.gz` has 4095 rows, since `cma` stopped on its own criteria one evaluation before the budget.

## Running your own optimizer

Any optimizer with an ask/tell interface can be run on a problem through the loop below. `spec` gives the frozen problem settings, `make_args` the frozen `TestAirfoils` arguments of its condition, `evaluate` the live XFOIL batch, and `screen` the stability gate. Run the loop from the repository root so that `bench.problems` and `airdbm_core` import.

~~~python
import numpy as np
from bench.problems import spec, make_args, evaluate, screen

s = spec("ADO-M-2-1")                       # frozen D, m, budget, condition
args = make_args(s["mach"], s["rey"])       # frozen TestAirfoils arguments
opt = Optimizer(n_var=s["ndim"], n_obj=s["m"])
history = []                                # every evaluation, in order
n_eval = 0
while n_eval < s["budget"]:
    X = opt.ask()                           # (n, D) in [0, 1]
    Y = evaluate(X, args)                   # live TestAirfoils batch
    Y = screen(X, Y, s["pid"], s["mach"], s["rey"], s["m"])  # gate
    opt.tell(X, Y)                          # maximize J1 (and J2)
    history.append(np.asarray(Y, float).reshape(len(X), -1))
    n_eval += len(X)
Y_run = np.vstack(history)[: s["budget"]]   # the first 1024·D evaluations
~~~

`Optimizer` stands for your solver. Both objectives are maximized, so a minimizer should be told `-Y`. For a single-objective problem (`ADO-S-*`), call `run_batch(X, args)` in place of `evaluate(X, args)`; `screen` then returns one column.

`screen` re-checks only the candidates the gate does not already account for, and those checks are extra solver calls outside the budget. The released panel searched on raw values and was screened afterward (`GATE = False` in the run scripts), so a run through this loop gets gated feedback during the search.

The loop evaluates entire batches, so the last batch can pass `1024·D`. The score keeps the first `1024·D` evaluations in evaluation order, as the released histories do. A full-budget run takes hours, so a short budget is a quick check of the wiring.

The score compares a run with the target in its `data.json`, in the same way as for the released panel:

~~~python
import json

def hv_2d(Y, ref):
    # dominated area above ref for two maximized objectives
    Y = Y[(Y[:, 0] > ref[0]) & (Y[:, 1] > ref[1])]
    Y = Y[np.argsort(-Y[:, 0])]
    area, top = 0.0, ref[1]
    for a, b in Y:
        if b > top:
            area += (a - ref[0]) * (b - top)
            top = b
    return area

d = json.load(open(f"bench/{s['pid']}/data.json"))
if s["m"] == 1:
    frac = Y_run[:, 0].max() / d["y_ref"]               # best Cl/Cd over the reference optimum
else:
    F = np.asarray(d["front_Y"])
    z = F.max(axis=0)                                  # each objective over its front maximum
    ref = (-0.05, -0.05)                               # 0.05 below the objective floor (0, 0)
    frac = hv_2d(Y_run / z, ref) / hv_2d(F / z, ref)   # normalized hypervolume ratio
~~~

For the released panel, the fraction of a method on a problem is the mean of this score over seeds 0 to 4.
