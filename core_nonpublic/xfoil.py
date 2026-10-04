import os
import shutil
import hashlib
import tempfile
import subprocess
import signal
import numpy as np
from scipy.signal import savgol_filter
from .constants import X_INTERP, XFOIL_APP, REYNOLDS, MACH, N_CRIT, CD_BRANCH_TOL, COARSE_ALPHA_STEP, REFINED_ALPHA_STEP, REFINE_ALPHA_WINDOW
from .polar import _compute_polar_metrics

def _resolve_xfoil_command(
    backend: str,
    apptainer_image: str,
    run_dir: str,
) -> tuple[list[str], str]:
    """
    Resolve command for XFOIL execution.
    Resolution order for auto mode is fixed: apptainer image, then system PATH 'xfoil'.
    """
    backend = (backend or 'auto').lower()
    if backend not in {'auto', 'native', 'apptainer'}:
        raise ValueError("xfoil_backend must be one of: 'auto', 'native', 'apptainer'.")

    def _apptainer_cmd() -> tuple[list[str], str]:
        apptainer_exe = shutil.which('apptainer')
        if not apptainer_exe:
            raise RuntimeError("Apptainer executable not found in system PATH.")

        image_abs = os.path.abspath(apptainer_image)
        bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), XFOIL_APP)
        if apptainer_image == XFOIL_APP and not os.path.exists(image_abs) and os.path.exists(bundled):
            image_abs = bundled
        if not os.path.exists(image_abs):
            raise RuntimeError(f"Apptainer image not found: {image_abs}")

        bind_arg = f"{run_dir}:{run_dir}"
        return [
            apptainer_exe,
            'run',
            '--bind', bind_arg,
            image_abs,
        ], 'apptainer'

    def _native_cmd() -> tuple[list[str], str]:
        system_xfoil = shutil.which('xfoil')
        if not system_xfoil:
            raise RuntimeError("XFOIL executable not found in system PATH.")
            
        xvfb_run = shutil.which('xvfb-run')
        if xvfb_run:
            return [
                xvfb_run, 
                '-a', 
                '-s', '-screen 0 640x480x8', 
                system_xfoil
            ], 'native-xvfb'
        else:
            return [system_xfoil], 'native-raw'

    if backend == 'apptainer':
        return _apptainer_cmd()

    if backend == 'native':
        return _native_cmd()

    try:
        return _apptainer_cmd()
    except Exception:
        return _native_cmd()

def _run_xfoil_aseq(
cmd: list[str],
    run_dir: str,
    coord_file: str,
    reynolds: float,
    mach: float,
    n_crit: float,
    repanel_n: int,
    max_iter: int,
    alpha_start: float,
    alpha_end: float,
    alpha_step: float,
    timeout_sec: float,
    warm_start: bool = False,
    anchor_alpha: float | None = None,
    diag: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    polar_stem = hashlib.sha256(os.urandom(32)).hexdigest()[:12]
    attempt_counter = {"n": 0}

    def _next_polar_name() -> str:
        attempt_counter["n"] += 1
        return f"p_{polar_stem}_{attempt_counter['n']}.out"

    coord_name = os.path.basename(coord_file)

    def _build_xfoil_input(use_ppar: bool, polar_name: str) -> str:
        commands = [f"LOAD {coord_name}"]

        if use_ppar and int(repanel_n) > 0:
            commands.extend(["PPAR", f"N {int(repanel_n)}", "", ""])
            
        commands.extend([
            "PANE",
            "OPER",
            f"VISC {reynolds:.0f}",
            f"MACH {mach:.6f}",
            "VPAR",
            f"N {n_crit:.6f}",
            "",
            f"ITER {max_iter}"
        ])

        if warm_start:
            start_ang = anchor_alpha if anchor_alpha is not None else 0.0
            commands.append(f"ALFA {start_ang:.3f}")
            
            gap = alpha_start - start_ang
            if abs(gap) > 0.5:
                step = 0.5 if gap > 0 else -0.5
                commands.append(f"ASEQ {start_ang:.3f} {alpha_start - step:.3f} {step:.3f}")

        commands.extend([
            "PACC",
            f"{polar_name}",
            "",
            f"ASEQ {alpha_start:.6f} {alpha_end:.6f} {alpha_step:.6f}",
            "PACC",
            "",
            "QUIT",
            ""
        ])
        
        return "\n".join(commands)

    def _run_once(exec_cmd: list[str], xfoil_input: str) -> subprocess.CompletedProcess:
        proc = subprocess.Popen(
            exec_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=run_dir,
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(
                input=xfoil_input, 
                timeout=timeout_sec if timeout_sec > 0.0 else None
            )
            return subprocess.CompletedProcess(proc.args, proc.returncode, stdout, stderr)
            
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            proc.communicate()
            raise
            
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except OSError:
                pass
            proc.communicate()
            raise

    polar_name = _next_polar_name()
    polar_path = os.path.join(run_dir, polar_name)

    try:
        proc = _run_once(cmd, _build_xfoil_input(use_ppar=True, polar_name=polar_name))

        if not os.path.exists(polar_path):
            if diag is not None:
                diag['ppar_fallbacks'] = int(diag.get('ppar_fallbacks', 0)) + 1
            polar_name = _next_polar_name()
            polar_path = os.path.join(run_dir, polar_name)
            proc = _run_once(cmd, _build_xfoil_input(use_ppar=False, polar_name=polar_name))

        if proc.returncode != 0 and diag is not None:
            diag['scan_aborts'] = int(diag.get('scan_aborts', 0)) + 1

    except subprocess.TimeoutExpired:
        if diag is not None:
            diag['scan_timeouts'] = int(diag.get('scan_timeouts', 0)) + 1
        e = np.array([])
        return e, e, e, e, e

    return _parse_xfoil_polar(polar_path)

def _write_airfoil_for_xfoil(airfoil: 'Airfoil', file_path: str) -> None:
    with open(file_path, 'w') as f:
        f.write(f"{airfoil.name}\n")
        for x, y in zip(X_INTERP, airfoil.colloc_vec):
            f.write(f"{x:.10f} {y:.10f}\n")

def _parse_xfoil_polar(polar_file: str) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                                  np.ndarray, np.ndarray]:
    """Read an XFOIL PACC polar as (alpha, Cl, Cd, Top_Xtr, Bot_Xtr)."""
    alpha_vals, cl_vals, cd_vals, xtr_top, xtr_bot = [], [], [], [], []
    if not os.path.exists(polar_file):
        e = np.array([])
        return e, e, e, e, e

    with open(polar_file, 'r') as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                alpha = float(parts[0])
                cl = float(parts[1])
                cd = float(parts[2])
            except ValueError:
                continue
            try:
                xt, xb = float(parts[5]), float(parts[6])
            except (ValueError, IndexError):
                xt, xb = float('nan'), float('nan')
            alpha_vals.append(alpha)
            cl_vals.append(cl)
            cd_vals.append(cd)
            xtr_top.append(xt)
            xtr_bot.append(xb)
    return (np.array(alpha_vals), np.array(cl_vals), np.array(cd_vals),
            np.array(xtr_top), np.array(xtr_bot))

def run_xfoil_evaluation(airfoil: 'Airfoil', xfoil_config: dict, m: int = 1) -> dict:
    """
    Execute XFOIL and compute polar metrics without truncating post-stall data.
    """
    apptainer_image = str(xfoil_config.get('apptainer_image', XFOIL_APP))
    backend = str(xfoil_config.get('xfoil_backend', 'auto'))

    reynolds = float(xfoil_config.get('reynolds', REYNOLDS))
    mach = float(xfoil_config.get('mach', MACH))
    n_crit = float(xfoil_config.get('n_crit', N_CRIT))
    
    repanel_n = int(xfoil_config.get('repanel_n', 160))
    max_iter = int(xfoil_config.get('xfoil_iter', 200))
    clcd_ceiling = float(xfoil_config.get('clcd_ceiling', 350.0))
    timeout_sec = float(xfoil_config.get('xfoil_timeout', 60.0))

    alpha_start = float(xfoil_config.get('alfa_start', 0.0))
    alpha_end = float(xfoil_config.get('alfa_end', 45.0))
    
    with tempfile.TemporaryDirectory(prefix='xfoil_run_') as run_dir:
        coord_file = os.path.join(run_dir, 'airfoil.dat')

        _write_airfoil_for_xfoil(airfoil, coord_file)
        cmd, runner = _resolve_xfoil_command(
            backend=backend,
            apptainer_image=apptainer_image,
            run_dir=run_dir,
        )

        scan_diag: dict = {}

        alpha_c, cl_c, cd_c, xt_c, xb_c = _run_xfoil_aseq(
            cmd=cmd,
            run_dir=run_dir,
            coord_file=coord_file,
            reynolds=reynolds,
            mach=mach,
            n_crit=n_crit,
            repanel_n=repanel_n,
            max_iter=max_iter,
            alpha_start=alpha_start,
            alpha_end=alpha_end,
            alpha_step=COARSE_ALPHA_STEP,
            timeout_sec=timeout_sec,
            warm_start=False,
            diag=scan_diag
        )

        valid_c = cd_c > 0
        refine_centers = []

        if np.any(valid_c):
            cl_cd_c = np.full_like(cl_c, np.nan, dtype=float)
            cl_cd_c[valid_c] = cl_c[valid_c] / cd_c[valid_c]
            
            cl_c_smooth = cl_c.copy()
            cl_cd_c_smooth = cl_cd_c.copy()
            
            valid_indices = np.where(valid_c)[0]
            if len(valid_indices) >= 5: 
                window_size = 5 
                poly_order = 3  
                cl_c_smooth[valid_indices] = savgol_filter(cl_c[valid_indices], window_size, poly_order)
                cl_cd_c_smooth[valid_indices] = savgol_filter(cl_cd_c[valid_indices], window_size, poly_order)

            alpha_center_clcd = float(alpha_c[int(np.nanargmax(cl_cd_c_smooth))])
            alpha_center_clmax = float(alpha_c[int(np.argmax(cl_c_smooth))])
            refine_centers = sorted({alpha_center_clcd, alpha_center_clmax})

        merged = {
            round(float(a), 6): (float(a), float(c_l), float(c_d), float(xt), float(xb))
            for a, c_l, c_d, xt, xb in zip(alpha_c, cl_c, cd_c, xt_c, xb_c)
        }

        refined_windows: list[tuple[float, float]] = []
        pending = list(refine_centers)
        MAX_EXTRA_REFINEMENTS = 3
        extra_used = 0

        def _covered(a: float) -> bool:
            return any(lo - 1e-9 <= a <= hi + 1e-9 for lo, hi in refined_windows)

        while pending:
            center = pending.pop(0)
            a0 = max(alpha_start, center - REFINE_ALPHA_WINDOW)
            a1 = min(alpha_end, center + REFINE_ALPHA_WINDOW)
            if a1 <= a0 or _covered(center):
                continue

            alpha_r, cl_r, cd_r, xt_r, xb_r = _run_xfoil_aseq(
                cmd=cmd,
                run_dir=run_dir,
                coord_file=coord_file,
                reynolds=reynolds,
                mach=mach,
                n_crit=n_crit,
                repanel_n=repanel_n,
                max_iter=max_iter,
                alpha_start=alpha_start,
                alpha_end=a1,
                alpha_step=REFINED_ALPHA_STEP,
                timeout_sec=timeout_sec,
                warm_start=False,
                diag=scan_diag
            )

            coarse_at = {round(float(a), 6): float(c) for a, c in zip(alpha_c, cd_c) if c > 0}
            mismatch = None
            for a, c_d in zip(alpha_r, cd_r):
                if c_d <= 0:
                    continue
                key = round(float(a), 6)
                c_ref = coarse_at.get(key)
                if c_ref is None or c_ref <= 0:
                    continue
                rel = abs(float(c_d) - c_ref) / c_ref
                if rel > CD_BRANCH_TOL:
                    mismatch = (float(a), c_ref, float(c_d), rel)
                    break

            if mismatch is not None:
                scan_diag['branch_mismatches'] = int(scan_diag.get('branch_mismatches', 0)) + 1
                continue

            for a, c_l, c_d, xt, xb in zip(alpha_r, cl_r, cd_r, xt_r, xb_r):
                merged[round(float(a), 6)] = (float(a), float(c_l), float(c_d),
                                              float(xt), float(xb))
            if len(alpha_r) > 0:
                refined_windows.append((alpha_start, a1))

            if not pending and extra_used < MAX_EXTRA_REFINEMENTS and merged:
                rows = sorted(merged.values(), key=lambda r: r[0])
                aa = np.array([r[0] for r in rows]); cc = np.array([r[1] for r in rows])
                dd = np.array([r[2] for r in rows])
                ok = dd > 0
                if np.any(ok):
                    ratio = np.full_like(cc, np.nan, dtype=float)
                    ratio[ok] = cc[ok] / dd[ok]
                    for peak_alpha in ({float(aa[int(np.nanargmax(ratio))]),
                                        float(aa[int(np.argmax(cc))])}):
                        if not _covered(peak_alpha):
                            pending.append(peak_alpha)
                            extra_used += 1
        if extra_used:
            scan_diag['extra_refinements'] = extra_used

        merged_rows = sorted(merged.values(), key=lambda row: row[0])
        alpha = np.array([r[0] for r in merged_rows], dtype=float)
        cl = np.array([r[1] for r in merged_rows], dtype=float)
        cd = np.array([r[2] for r in merged_rows], dtype=float)
        xtr_top = np.array([r[3] for r in merged_rows], dtype=float)
        xtr_bot = np.array([r[4] for r in merged_rows], dtype=float)

        if len(alpha) == 0:
            metrics = {
                'alpha': [],
                'cl': [],
                'cd': [],
                'cl_cd': [],
                'cl_cd_max': 0.0,          # Worst possible efficiency
                'alpha_at_cl_cd_max': 0.0,
                'cl_max': 0.0,             # Worst possible lift
                'alpha_at_cl_max': 0.0,
                'alpha_stall': 0.0,
                'delta_alpha': 0.0,        # Worst possible stall margin
                'error': 'All XFOIL scans failed to converge.'
            }
        else:
            metrics = _compute_polar_metrics(alpha, cl, cd, reynolds,
                                             clcd_ceiling=clcd_ceiling,
                                             xtr_top=xtr_top, xtr_bot=xtr_bot)

        metrics['runner'] = runner
        metrics['command'] = cmd
        metrics['scan_scheme'] = {
            'coarse_step': COARSE_ALPHA_STEP,
            'refined_step': REFINED_ALPHA_STEP,
            'refine_window': REFINE_ALPHA_WINDOW,
        }
        metrics['scan_timeouts'] = int(scan_diag.get('scan_timeouts', 0))
        metrics['extra_refinements'] = int(scan_diag.get('extra_refinements', 0))
        metrics['ppar_fallbacks'] = int(scan_diag.get('ppar_fallbacks', 0))
        metrics['branch_mismatches'] = int(scan_diag.get('branch_mismatches', 0))
        metrics['scan_aborts'] = int(scan_diag.get('scan_aborts', 0))
        
        return metrics
