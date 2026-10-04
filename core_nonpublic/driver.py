import numpy as np
from .airfoil import Airfoil
from .geometry import create_morphed_airfoil
from .database import get_baselines
from .polar import _extract_objectives_from_xfoil_result
from .xfoil import run_xfoil_evaluation

def _evaluate_single_candidate(
    phi: np.ndarray,
    weight_range: tuple[float, float],
    normalization: str | None,
    data_folder: str,
    selected_baseline_names: list[str],
    xfoil_config: dict,
    m : int,
) -> Airfoil:
    """Worker-safe single-candidate evaluator for parallel TestAirfoils execution."""
    baselines = get_baselines(data_folder, selected_baseline_names)
    w_lower, w_upper = weight_range

    weights = w_lower + phi * (w_upper - w_lower)

    if normalization == 'SUM':
        norm = np.sum(weights)
        if norm > 1e-8:
            weights = weights / norm
        else:
            raise ValueError("Sum of weights is too close to zero for normalization.")
    elif normalization == 'ABS_SUM':
        norm = np.sum(np.abs(weights))
        if norm > 1e-8:
            weights = weights / norm
        else:
            raise ValueError("Sum of absolute weights is too close to zero for normalization.")

    morphed_airfoil = create_morphed_airfoil(weights, baselines, correct_geometry=True)
    morphed_airfoil.name = "_".join(f"{w:.2f}" for w in weights)[:70]

    if xfoil_config.get('xfoil_evaluation', False):
        strict = bool(xfoil_config.get('xfoil_strict', True))
        retry_count = max(0, int(xfoil_config.get('xfoil_retry', 1)))

        def _is_empty_polar(metrics: dict | None) -> bool:
            if not metrics:
                return True
            return (
                len(metrics.get('alpha', [])) == 0
                and len(metrics.get('cl', [])) == 0
                and len(metrics.get('cd', [])) == 0
            )

        last_exc = None
        for attempt in range(retry_count + 1):
            try:
                metrics = run_xfoil_evaluation(morphed_airfoil, xfoil_config, m)
                incomplete = _is_empty_polar(metrics) or int(metrics.get('scan_timeouts', 0) or 0) > 0
                if incomplete and attempt < retry_count:
                    continue
                morphed_airfoil.xfoil_result = metrics
                break
            except Exception as exc:
                last_exc = exc
                if attempt < retry_count:
                    continue
                if strict:
                    raise
                morphed_airfoil.xfoil_result = {
                    'error': str(exc),
                    'runner': None,
                }

        if morphed_airfoil.xfoil_result is None and last_exc is not None and not strict:
            morphed_airfoil.xfoil_result = {
                'error': str(last_exc),
                'runner': None,
            }

    return morphed_airfoil

class AirfoilEvaluationResult:
    """Container for a generated airfoil and optional XFOIL-derived outputs."""

    def __init__(
        self,
        airfoil: Airfoil,
        xfoil_result: dict | None = None,
        objectives: float | list[float] | None = None,
    ):
        self.airfoil = airfoil
        self.xfoil_result = xfoil_result
        self.objectives = objectives


def _format_testairfoils_output(
    airfoils: list[Airfoil],
    xfoil_evaluation: bool,
    m: int,
) -> list[AirfoilEvaluationResult]:
    """
    Always return container objects so callers can access generated airfoils.
    When XFOIL is enabled, objective values are populated from attached XFOIL results.
    """
    outputs: list[AirfoilEvaluationResult] = []
    for airfoil in airfoils:
        xr = airfoil.xfoil_result if xfoil_evaluation else None
        objectives = None
        if xfoil_evaluation:
            objectives = _extract_objectives_from_xfoil_result(xr or {}, m)

        outputs.append(AirfoilEvaluationResult(airfoil=airfoil, xfoil_result=xr, objectives=objectives))

    return outputs
