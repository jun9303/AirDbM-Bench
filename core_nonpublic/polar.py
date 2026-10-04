import numpy as np
from scipy.signal import savgol_filter
from .constants import CD_DROP_TOL, CD_BUCKET_DROP_TOL, LAMINAR_XTR

def _compute_polar_metrics(
    alpha: np.ndarray,
    cl: np.ndarray,
    cd: np.ndarray,
    reynolds: float,
    clcd_ceiling: float = 350.0,
    xtr_top: np.ndarray | None = None,
    xtr_bot: np.ndarray | None = None,
) -> dict:
    cd_min_physical = 2.656 / np.sqrt(reynolds)
    valid = cd > cd_min_physical

    CD_RISE_EPS = 0.02   # a rise of >2% marks the bucket as passed (ignores numerical jitter)
    risen = False
    for i in range(1, len(cd)):
        if cd[i - 1] <= 0 or cd[i] <= 0:
            continue
        if cd[i] > cd[i - 1] * (1.0 + CD_RISE_EPS):
            risen = True              # past the drag bucket: drag is now increasing with alpha
        elif risen and cd[i] < cd[i - 1] * (1.0 - CD_DROP_TOL):
            valid[i:] = False         # branch jump: this point and all beyond are off-branch
            break
        elif not risen and cd[i] < cd[i - 1] * (1.0 - CD_BUCKET_DROP_TOL):
            valid[i:] = False
            break

    if not np.any(valid):
        return {
            'alpha': alpha.tolist(),
            'cl': cl.tolist(),
            'cd': cd.tolist(),
            'cl_cd': cl.tolist(),
            'cl_cd_max': 0.0,
            'alpha_at_cl_cd_max': 0.0,
            'cl_max': 0.0,
            'alpha_at_cl_max': 0.0,
            'alpha_stall': 0.0,
            'delta_alpha': 0.0,
            'xtr_top': [] if xtr_top is None else np.asarray(xtr_top).tolist(),
            'xtr_bot': [] if xtr_bot is None else np.asarray(xtr_bot).tolist(),
            'error': f'All valid Cd values were below physical threshold or solver continuity broke.'
        }

    cl_cd = np.full_like(cl, np.nan, dtype=float)
    cl_cd[valid] = cl[valid] / cd[valid]

    cl_cd = np.clip(cl_cd, a_min=None, a_max=float(clcd_ceiling))

    cl_smooth = cl.copy()
    cl_cd_smooth = cl_cd.copy()
    valid_indices = np.where(valid)[0]
    if len(valid_indices) >= 5:
        window_size = min(5, len(valid_indices))
        if window_size % 2 == 0:
            window_size -= 1
        poly_order = min(3, window_size - 1)

        if window_size >= 3 and poly_order >= 1:
            cl_smooth[valid_indices] = savgol_filter(cl[valid_indices], window_size, poly_order)
            cl_cd_smooth[valid_indices] = savgol_filter(cl_cd[valid_indices], window_size, poly_order)

    cl_valid = np.where(valid, cl_smooth, np.nan)
    idx_best = int(np.nanargmax(cl_cd_smooth))
    alpha_best = float(alpha[idx_best])
    idx_cl_max = int(np.nanargmax(cl_valid))

    idx_stall = None
    if cl_smooth.size >= 3:
        start_idx = max(1, idx_best)
        for i in range(start_idx, len(cl_smooth) - 1):
            if not (valid[i - 1] and valid[i] and valid[i + 1]):
                continue
            if cl_smooth[i] > cl_smooth[i - 1] and cl_smooth[i] >= cl_smooth[i + 1]:
                idx_stall = i
                break
    if idx_stall is None:
        idx_stall = max(idx_best, idx_cl_max)

    alpha_stall = float(alpha[idx_stall])
    raw_delta_alpha = float(alpha_stall - alpha_best)
    delta_alpha = raw_delta_alpha if raw_delta_alpha > 1e-9 else 0.0

    out = {
        'alpha': alpha.tolist(),
        'cl': cl.tolist(),
        'cd': cd.tolist(),
        'cl_cd': cl_cd.tolist(),
        'cl_cd_max': float(cl_cd[idx_best]),
        'alpha_at_cl_cd_max': alpha_best,
        'cl_max': float(cl[idx_cl_max]),
        'alpha_at_cl_max': float(alpha[idx_cl_max]),
        'alpha_stall': alpha_stall,
        'delta_alpha': delta_alpha,
    }
    if xtr_top is not None and xtr_bot is not None and len(xtr_top) == len(alpha):
        xt = np.asarray(xtr_top, dtype=float)
        xb = np.asarray(xtr_bot, dtype=float)
        i0 = int(np.argmin(np.abs(alpha)))          # the alpha = 0 row of the same march
        out['xtr_top'] = xt.tolist()
        out['xtr_bot'] = xb.tolist()
        out['xtr_top_at_zero'] = float(xt[i0])
        out['xtr_bot_at_zero'] = float(xb[i0])
        out['xtr_top_at_peak'] = float(xt[idx_best])
        out['xtr_bot_at_peak'] = float(xb[idx_best])
        both = max(float(xt[i0]), float(xb[i0]))
        out['fully_laminar_at_zero'] = bool(np.isfinite(both)
                                            and min(float(xt[i0]), float(xb[i0])) >= LAMINAR_XTR)
    return out


def _extract_objectives_from_xfoil_result(xfoil_result: dict, m: int) -> float | list[float]:
    if m not in (1, 2):
        raise ValueError("When xfoil_evaluation=True, only m=1 or m=2 are currently supported.")

    cl_cd_max_raw = xfoil_result.get('cl_cd_max', np.nan)
    delta_alpha_raw = xfoil_result.get('delta_alpha', np.nan)

    cl_cd_max = np.nan if cl_cd_max_raw is None else float(cl_cd_max_raw)
    delta_alpha = np.nan if delta_alpha_raw is None else float(delta_alpha_raw)

    if not np.isnan(delta_alpha):
        delta_alpha = delta_alpha if delta_alpha > 0.0 else 0.0

    if m == 1:
        return cl_cd_max
    return [cl_cd_max, delta_alpha]
