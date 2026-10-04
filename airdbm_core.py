"""
dbm_api.py
API script for DbM Airfoil generation and evaluation.
"""

import os
import hashlib
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from core_nonpublic.constants import DATA_FOLDER, EXPECTED_BASELINES, XFOIL_APP, REYNOLDS, MACH, N_CRIT
from core_nonpublic.driver import _evaluate_single_candidate, AirfoilEvaluationResult, _format_testairfoils_output
from core_nonpublic.constants import NUM_POINTS_INTERP, THETA_HALF, X_INTERP_HALF_ASC, X_INTERP_HALF_DESC, X_INTERP, MIN_INTERIOR_THICKNESS, MIN_TRAILING_EDGE_THICKNESS, CD_BRANCH_TOL, CD_DROP_TOL, CD_BUCKET_DROP_TOL, COARSE_ALPHA_STEP, REFINED_ALPHA_STEP, REFINE_ALPHA_WINDOW, LAMINAR_XTR
from core_nonpublic.airfoil import Airfoil

__version__ = "1.0"#"0.2.1"

def TestAirfoils(x: np.ndarray, args: dict = None, m: int = 2) -> list[AirfoilEvaluationResult]:
    """
    Test function for generating DbM airfoils based on an N x D candidate matrix.

        Parameters:
        - x: N x D array, where N is candidates and D is design parameters.
                 Input values must range from 0 to 1.
        - args: Configuration dictionary supporting:
            AIRFOIL DESIGN ARGS:
                - 'airfoil_db_dir': Path to airfoil database (defaults to DATA_FOLDER).
                - 'dbm_baselines': List of airfoil names to use as baselines (defaults to EXPECTED_BASELINES).
                - 'dbm_weight_range': [lower, upper] limits (default: [-1.0, 1.0]). Allows negative weights for extrapolative morphing.
                - 'dbm_normalization': None (default), 'SUM', or 'ABS_SUM'.
            AIRFOIL EVALUATION ARGS:
                - 'xfoil_evaluation': Run XFOIL evaluation after geometry generation (default: True).
                - 'xfoil_backend': 'apptainer' (default), 'auto', or 'native'.
                    ('auto': apptainer image first, then system 'xfoil').
                - 'apptainer_image': Path to apptainer image containing xfoil (default: XFOIL_APP).
                - 'xfoil_iter': Max XFOIL iterations per alpha (default: 200).
                - 'xfoil_timeout': Timeout in seconds per candidate (default: 60.0).
                - 'xfoil_retry': Number of reattempts when no polar points are parsed at all (default: 1).
                    Useful for transient XFOIL deadlock/no-parse cases that can occur in multiprocessing,
                    even when the design itself converges in other runs.
                - 'xfoil_strict': If True, fail on XFOIL errors; if False, attach error in airfoil.xfoil_result.
                - 'alfa_start', 'alfa_end': Polar angle range (defaults: 0, 45).
                - 'reynolds': Reynolds number (defaults to REYNOLDS).
                - 'mach': Mach number for XFOIL (default: MACH=0).
                - 'n_crit': e^N critical amplification factor (default: N_CRIT=9).
                - 'repanel_n': Internal XFOIL repanel node count via PPAR/N (default: 160).
            MULTIPROCESSING ARGS:
                - 'parallel': Enable multiprocessing across candidates (default: True).
                - 'max_workers': Max worker processes (default: total CPU count in environment).
    - m: Number of objectives (1 for Cl/Cd <single objective>, 2 for Cl/Cd and delta alpha <bi-objective>).

    Returns:
    - A list of AirfoilEvaluationResult objects of length N.
      Each element provides:
        - .airfoil: generated Airfoil object (always available)
        - .xfoil_result: raw XFOIL metrics dict when xfoil_evaluation=True, else None
        - .objectives:
            - None when xfoil_evaluation=False
            - m=1 -> Cl/Cd_max (float)
            - m=2 -> [Cl/Cd_max, delta_alpha]
    """
    if args is None:
        args = {}
        
    data_folder = args.get('airfoil_db_dir', DATA_FOLDER)
    weight_range = args.get('dbm_weight_range', [-1.0, 1.0])
    normalization = args.get('dbm_normalization', None)
    expected_baselines = args.get('dbm_baselines', EXPECTED_BASELINES)
    parallel = args.get('parallel', True)
    try:
        cpu_total = max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        cpu_total = max(1, os.cpu_count() or 1)
    max_workers_cfg = args.get('max_workers', cpu_total)
    max_workers = min(max(1, int(max_workers_cfg)), cpu_total)
    xfoil_config = {
        'xfoil_evaluation': args.get('xfoil_evaluation', True),
        'xfoil_backend': args.get('xfoil_backend', 'apptainer'),
        'apptainer_image': args.get('apptainer_image', XFOIL_APP),
        'reynolds': args.get('reynolds', REYNOLDS),
        'mach': args.get('mach', MACH),
        'n_crit': args.get('n_crit', N_CRIT),
        'alfa_start': args.get('alfa_start', 0.0),
        'alfa_end': args.get('alfa_end', 45.0),
        'repanel_n': args.get('repanel_n', 160),
        'xfoil_iter': args.get('xfoil_iter', 200),
        'xfoil_timeout': args.get('xfoil_timeout', 60.0),
        'xfoil_retry': args.get('xfoil_retry', 1),
        'xfoil_strict': args.get('xfoil_strict', True),
        'clcd_ceiling': args.get('clcd_ceiling', 350.0),
    }
    
    x = np.atleast_2d(x)

    N, D = x.shape
    
    if D > len(expected_baselines):
        raise ValueError(f"Number of design parameters (D={D}) exceeds the number of provided baselines ({len(expected_baselines)}).")
    
    selected_baseline_names = expected_baselines[:D]

    w_lower, w_upper = float(weight_range[0]), float(weight_range[1])
    if w_lower > 0.0 or w_upper < 1.0:
        raise ValueError(
            f"Invalid dbm_weight_range: [{w_lower}, {w_upper}]. "
            "The range must fully encompass [0.0, 1.0] to guarantee that "
            "pure baseline reconstruction remains mathematically possible."
        )
    weight_range_tuple = (w_lower, w_upper)
    xfoil_evaluation = bool(xfoil_config.get('xfoil_evaluation', True))

    if parallel and N > 1 and max_workers > 1:
        workers = min(max_workers, N)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            airfoils = list(
                executor.map(
                    _evaluate_single_candidate,
                    x,
                    repeat(weight_range_tuple),
                    repeat(normalization),
                    repeat(data_folder),
                    repeat(selected_baseline_names),
                    repeat(xfoil_config),
                    repeat(m),
                )
            )
        return _format_testairfoils_output(airfoils, xfoil_evaluation=xfoil_evaluation, m=m)

    airfoils = [
        _evaluate_single_candidate(
            phi=x[i, :],
            weight_range=weight_range_tuple,
            normalization=normalization,
            data_folder=data_folder,
            selected_baseline_names=selected_baseline_names,
            xfoil_config=xfoil_config,
            m=m,
        )
        for i in range(N)
    ]
    return _format_testairfoils_output(airfoils, xfoil_evaluation=xfoil_evaluation, m=m)

def VerifyDesigns(
    x: np.ndarray,
    args: dict = None,
    m: int = 2,
    eps: float = 1e-6,
    n_dir: int = 4,
    rel_tol: float = 0.02,
    objective: int = 0,
    directions: str = "axes",
    seed: int = 0,
) -> list[dict]:
    """Perturbation test for designs that are about to be published as reference solutions."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    N, D = x.shape
    if n_dir < 1:
        raise ValueError("n_dir must be at least 1.")
    if not 0 <= objective < m:
        raise ValueError(f"objective index {objective} outside the {m} objectives.")
    if directions == "axes":
        unit = np.concatenate([np.eye(D), -np.eye(D)], axis=0)      # (2D, D)
        dirs = np.broadcast_to(unit, (N, 2 * D, D))
        n_probe = 2 * D
    elif directions == "random":
        def _dirs(xi):
            h = hashlib.sha256(np.ascontiguousarray(xi, dtype=np.float64).tobytes()
                               + str(int(seed)).encode()).digest()
            g = np.random.default_rng(int.from_bytes(h[:8], "little"))
            d = g.normal(size=(n_dir, len(xi)))
            return d / np.linalg.norm(d, axis=1, keepdims=True)

        dirs = np.stack([_dirs(xi) for xi in x])
        n_probe = n_dir
    else:
        raise ValueError("directions must be 'axes' or 'random'")
    batch = [x]
    for k in range(n_probe):
        batch.append(np.clip(x + eps * dirs[:, k, :], 0.0, 1.0))
    Y = np.array([
        np.atleast_1d(np.asarray(r.objectives, dtype=float))
        for r in TestAirfoils(np.vstack(batch), args=args, m=m)
    ]).reshape(n_probe + 1, N, -1)

    out = []
    for i in range(N):
        y0, yn = Y[0, i], Y[1:, i]
        keep = yn[:, objective] > 0.0
        info = yn[keep]
        med = np.median(info, axis=0) if len(info) else np.full_like(y0, np.nan)
        dev = (np.abs(med - y0) / np.maximum(np.abs(y0), 1e-12) if len(info)
               else np.full_like(y0, np.nan))
        q = (np.percentile(np.abs(info[:, objective] - y0[objective])
                           / max(abs(y0[objective]), 1e-12), [0, 25, 50, 75, 100]).tolist()
             if len(info) else [float('nan')] * 5)
        out.append({
            'objectives': y0.tolist() if m > 1 else float(y0[0]),
            'neighbors': yn.tolist(),
            'median_neighbor': med.tolist(),
            'rel_dev': dev.tolist(),
            'abs_dev': (np.abs(med - y0)).tolist(),
            'rel_dev_quantiles': q,
            'frac_within_tol': (float(np.mean(np.abs(info[:, objective] - y0[objective])
                                              <= rel_tol * max(abs(y0[objective]), 1e-12)))
                                if len(info) else float('nan')),
            'directions': directions,
            'n_probes': int(n_probe),
            'n_informative': int(keep.sum()),
            'robust': bool(keep.sum() >= 2 and dev[objective] <= rel_tol),
        })
    return out
