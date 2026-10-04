import numpy as np
from scipy.interpolate import PchipInterpolator
from .constants import NUM_POINTS_INTERP, X_INTERP, MIN_INTERIOR_THICKNESS, MIN_TRAILING_EDGE_THICKNESS
from .airfoil import Airfoil

try:
    from shapely.geometry import Polygon, LineString, Point, MultiPoint
    from shapely.validation import make_valid
except ImportError:
    Polygon = None
    print("Warning: Shapely library not available. Geometry correction will be bypassed.")

def moving_average(y_values: np.ndarray, window_size: int = 5) -> np.ndarray:
    if window_size < 1: return y_values
    return np.convolve(y_values, np.ones(window_size)/window_size, mode='same')

def _smooth_surface_preserve_endpoints(y_values: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Smooth a 1D surface while preserving endpoint values (LE/TE anchors)."""
    if window_size <= 1 or y_values.size < 3:
        return y_values.copy()

    if window_size % 2 == 0:
        window_size += 1

    pad = window_size // 2
    padded = np.pad(y_values, (pad, pad), mode='edge')
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode='valid')

    smoothed[0] = y_values[0]
    smoothed[-1] = y_values[-1]
    return smoothed

def _prepare_surface_for_interp(x_surface: np.ndarray, y_surface: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Sort a surface in ascending x and merge duplicate x values using y-averaging.
    Returns None when insufficient points remain for interpolation.
    """
    if len(x_surface) < 2:
        return None

    order = np.argsort(x_surface)
    x_sorted = x_surface[order]
    y_sorted = y_surface[order]

    x_unique, inv = np.unique(x_sorted, return_inverse=True)
    if x_unique.size < 2:
        return None

    y_acc = np.zeros_like(x_unique, dtype=float)
    counts = np.zeros_like(x_unique, dtype=float)
    for i, grp in enumerate(inv):
        y_acc[grp] += y_sorted[i]
        counts[grp] += 1.0
    y_unique = y_acc / counts

    return x_unique, y_unique

def _resample_surface(x_surface: np.ndarray, y_surface: np.ndarray, x_target: np.ndarray) -> np.ndarray | None:
    prepared = _prepare_surface_for_interp(x_surface, y_surface)
    if prepared is None:
        return None
    x_unique, y_unique = prepared

    try:
        interpolator = PchipInterpolator(x_unique, y_unique)
    except ValueError:
        return None

    return interpolator(x_target)

def _enforce_min_interior_thickness(
    y_upper: np.ndarray,
    y_lower: np.ndarray,
    min_thickness: float = MIN_INTERIOR_THICKNESS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Enforce minimum thickness on interior points only (exclude LE/TE anchors).
    This prevents local surface crossing and near-zero-thickness pockets.
    """
    y_u = y_upper.copy()
    y_l = y_lower.copy()

    if y_u.size < 3 or y_l.size < 3:
        return y_u, y_l

    thickness = y_u - y_l
    for i in range(1, len(thickness) - 1):
        if thickness[i] < min_thickness:
            delta = 0.5 * (min_thickness - thickness[i])
            y_u[i] += delta
            y_l[i] -= delta

    return y_u, y_l

def interp_airfoil(x_coords: np.ndarray, y_coords: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    if len(x_coords) != len(y_coords) or len(x_coords) < 4:
        return None

    min_x = np.min(x_coords)
    le_indices = np.where(np.isclose(x_coords, min_x, atol=1e-8))[0]
    if le_indices.size == 0:
        return None

    le_start, le_end = int(le_indices[0]), int(le_indices[-1])
    x_upper_raw, y_upper_raw = x_coords[:le_start + 1].copy(), y_coords[:le_start + 1].copy()
    x_lower_raw, y_lower_raw = x_coords[le_end:].copy(), y_coords[le_end:].copy()

    if len(x_lower_raw) < 2 or len(x_upper_raw) < 2:
        return None

    y_le = np.mean(y_coords[le_indices])
    x_upper_raw[-1], y_upper_raw[-1] = min_x, y_le
    x_lower_raw[0], y_lower_raw[0] = min_x, y_le

    prepared_upper = _prepare_surface_for_interp(x_upper_raw, y_upper_raw)
    prepared_lower = _prepare_surface_for_interp(x_lower_raw, y_lower_raw)
    if prepared_upper is None or prepared_lower is None:
        return None

    x_upper, y_upper = prepared_upper
    x_lower, y_lower = prepared_lower

    try:
        f_upper = PchipInterpolator(x_upper, y_upper)
        f_lower = PchipInterpolator(x_lower, y_lower)
    except ValueError:
        return None

    y_interp_upper = f_upper(X_INTERP[:NUM_POINTS_INTERP // 2 + 1])
    y_interp_lower = f_lower(X_INTERP[NUM_POINTS_INTERP // 2:])

    y_le_interp = 0.5 * (y_interp_upper[-1] + y_interp_lower[0])
    y_interp_upper[-1] = y_le_interp
    y_interp_lower[0] = y_le_interp

    y_interp_combined = np.concatenate((y_interp_upper, y_interp_lower[1:]))
    return X_INTERP, y_interp_combined

def correct_airfoil_geometry(airfoil_to_correct: Airfoil) -> Airfoil:
    if Polygon is None:
        return airfoil_to_correct

    points = list(zip(X_INTERP, airfoil_to_correct.colloc_vec))
    if tuple(points[0]) != tuple(points[-1]):
        points.append(points[0])

    try:
        airfoil_shape_current = Polygon(points).simplify(1e-5, preserve_topology=True)
    except Exception:
        return airfoil_to_correct

    try:
        airfoil_shape_current = make_valid(airfoil_shape_current)
        if not isinstance(airfoil_shape_current, Polygon):
            if hasattr(airfoil_shape_current, 'geoms'):
                polygons = [g for g in airfoil_shape_current.geoms if isinstance(g, Polygon)]
                if polygons: airfoil_shape_current = max(polygons, key=lambda p: p.area)
                else: return airfoil_to_correct
            else: return airfoil_to_correct

        x_slices = X_INTERP[NUM_POINTS_INTERP // 2:]
        upper_surface_pts, lower_surface_pts = [], []
        final_boundary = LineString(airfoil_shape_current.exterior.coords)

        for x_val in x_slices:
            vertical_line = LineString([(x_val, -1), (x_val, 1)])
            intersection = final_boundary.intersection(vertical_line)
            if intersection.is_empty: continue
            
            y_at_x = []
            if isinstance(intersection, Point): y_at_x.append(intersection.y)
            elif hasattr(intersection, 'geoms'):
                for geom_item in intersection.geoms:
                    if isinstance(geom_item, Point): y_at_x.append(geom_item.y)

            if not y_at_x: continue
            y_at_x.sort(reverse=True)
            upper_surface_pts.append((x_val, y_at_x[0]))
            lower_surface_pts.append((x_val, y_at_x[-1]))
        
        if not upper_surface_pts or not lower_surface_pts: return airfoil_to_correct

        x_upper_hit = np.array([p[0] for p in upper_surface_pts])
        y_upper_hit = np.array([p[1] for p in upper_surface_pts])
        x_lower_hit = np.array([p[0] for p in lower_surface_pts])
        y_lower_hit = np.array([p[1] for p in lower_surface_pts])

        y_upper_resampled = _resample_surface(x_upper_hit, y_upper_hit, x_slices)
        y_lower_resampled = _resample_surface(x_lower_hit, y_lower_hit, x_slices)
        if y_upper_resampled is None or y_lower_resampled is None:
            return airfoil_to_correct

        y_upper_smoothed = _smooth_surface_preserve_endpoints(y_upper_resampled, window_size=5)
        y_lower_smoothed = _smooth_surface_preserve_endpoints(y_lower_resampled, window_size=5)

        y_le = 0.5 * (y_upper_smoothed[0] + y_lower_smoothed[0])
        y_upper_smoothed[0] = y_le
        y_lower_smoothed[0] = y_le

        y_upper_smoothed, y_lower_smoothed = _enforce_min_interior_thickness(
            y_upper_smoothed,
            y_lower_smoothed,
            min_thickness=MIN_INTERIOR_THICKNESS,
        )

        te_thickness = y_upper_smoothed[-1] - y_lower_smoothed[-1]
        if te_thickness < MIN_TRAILING_EDGE_THICKNESS:
            te_delta = 0.5 * (MIN_TRAILING_EDGE_THICKNESS - te_thickness)
            y_upper_smoothed[-1] += te_delta
            y_lower_smoothed[-1] -= te_delta

        y_selig_final = np.concatenate((y_upper_smoothed[::-1], y_lower_smoothed[1:]))

        return Airfoil(
            airfoil_id=airfoil_to_correct.airfoil_id,
            name=f"{airfoil_to_correct.name}_corrected",
            colloc_vec=y_selig_final,
            x_raw=X_INTERP,
            y_raw=y_selig_final.copy()
        )
    except Exception:
        return airfoil_to_correct

def create_morphed_airfoil(weights: np.ndarray, baseline_airfoils: list[Airfoil], correct_geometry: bool = True) -> Airfoil:
    if len(weights) != len(baseline_airfoils):
        raise ValueError("Length of weights must match the number of baseline airfoils.")

    weights_np = np.array(weights)
    
    
    if np.allclose(weights_np, 0):
        return Airfoil(airfoil_id=-1, name='Morphed_FlatLine', colloc_vec=np.zeros_like(X_INTERP),
                       x_raw=X_INTERP.copy(), y_raw=np.zeros_like(X_INTERP))

    y_morphed_sum = np.zeros_like(X_INTERP)
    for airfoil, weight in zip(baseline_airfoils, weights_np):
        y_morphed_sum += weight * airfoil.get_interpolated_data()
    
    weight_str = "_".join(f"{w:.4f}" for w in weights_np)
    morphed_name = f"Morphed_W[{weight_str[:50]}]"

    morphed_obj = Airfoil(airfoil_id=-1, name=morphed_name, colloc_vec=y_morphed_sum,
                          x_raw=X_INTERP.copy(), y_raw=y_morphed_sum.copy())
    
    if correct_geometry:
        return correct_airfoil_geometry(morphed_obj)
    return morphed_obj
