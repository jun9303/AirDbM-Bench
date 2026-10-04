import os
import pickle
import tempfile
import numpy as np
from .airfoil import Airfoil
from .constants import DATA_FOLDER
from .geometry import interp_airfoil

_CACHED_AIRFOIL_DB_DICT = None

def read_airfoil_file(file_path: str) -> tuple[str, np.ndarray, np.ndarray]:
    with open(file_path, 'r') as f:
        lines = f.readlines()
        name = lines[0].strip()
        coords_list = []
        is_goe451 = name.startswith("GOE 451")

        for line_content in lines[1:]:
            line_content = line_content.strip()
            if line_content.startswith("#") or not line_content:
                continue
            coords_list.append(list(map(float, line_content.split())))
        
        coords_np = np.array(coords_list)
        x_original, y_original = coords_np[:, 0], coords_np[:, 1]

        if is_goe451: # DB has error for this specific airfoil
            y_original[x_original == 0.0249500] = 0.0192620 

        x_min, x_max = x_original.min(), x_original.max()
        if x_max > x_min:
            x_original = (x_original - x_min) / (x_max - x_min)

    leading_edge_indices = np.where(np.isclose(x_original, 0, atol=1e-8))[0]

    if len(leading_edge_indices) == 0:
        for i in range(1, len(x_original)):
            if x_original[i-1] > 0 and x_original[i] < 0:
                y_le = np.interp(0, [x_original[i-1], x_original[i]], [y_original[i-1], y_original[i]])
                x_original = np.insert(x_original, i, 0)
                y_original = np.insert(y_original, i, y_le)
                break
    elif len(leading_edge_indices) == 2:
        upper_le_idx, lower_le_idx = leading_edge_indices
        y_avg_le = (y_original[upper_le_idx] + y_original[lower_le_idx]) / 2.0
        x_original[upper_le_idx] = 1e-6 
        x_original[lower_le_idx] = 1e-6
        insert_idx = min(upper_le_idx, lower_le_idx) + 1
        x_original = np.insert(x_original, insert_idx, 0)
        y_original = np.insert(y_original, insert_idx, y_avg_le)
        
    return name, x_original, y_original

def _load_cached_db(pickle_file_path: str) -> list['Airfoil'] | None:
    """
    Load the cached database pickle, returning None when it is missing, empty,
    or corrupt so the caller can rebuild from the raw .dat files instead.

    Under multiprocessing, a pickle can be observed while another worker (or a
    packaged-data copy step) is still writing it, yielding a partially written
    file. Reading such a file raises 'pickle data was truncated' / EOFError.
    Treating any unreadable pickle as a cache miss keeps parallel runs robust.
    """
    if not os.path.exists(pickle_file_path):
        return None
    try:
        if os.path.getsize(pickle_file_path) == 0:
            return None
        with open(pickle_file_path, 'rb') as f:
            db = pickle.load(f)
    except Exception:
        return None
    if not isinstance(db, list) or not db:
        return None
    return db


def _build_db_from_dat_files(data_folder: str) -> tuple[list['Airfoil'], bool]:
    """
    Rebuild the airfoil database directly from Selig-format .dat files.

    Returns (airfoils_db, complete). 'complete' is False when a .dat file could
    not be read (e.g. it is still being copied by a concurrent packaged-data
    populate step); the caller then avoids persisting a partial cache.
    """
    airfoils_db: list['Airfoil'] = []
    if not os.path.isdir(data_folder):
        return airfoils_db, True

    incompatible_files = {'30p-30n.dat', 'naca1.dat'}
    affile_list_filtered = [
        f for f in os.listdir(data_folder)
        if f not in incompatible_files and f.endswith('.dat')
    ]

    complete = True
    for idx, filename in enumerate(sorted(affile_list_filtered)):
        file_full_path = os.path.join(data_folder, filename)
        try:
            af_name, x_o, y_o = read_airfoil_file(file_full_path)
        except (OSError, ValueError, IndexError):
            complete = False
            continue

        interp_result = interp_airfoil(x_o, y_o)
        if interp_result is None:
            continue

        _, y_interp = interp_result
        airfoils_db.append(
            Airfoil(airfoil_id=idx, name=af_name, colloc_vec=y_interp, x_raw=x_o, y_raw=y_o)
        )

    return airfoils_db, complete


def _atomic_pickle_dump(obj: object, pickle_file_path: str) -> None:
    """
    Write a pickle atomically so concurrent readers never observe a partial file.

    The object is written to a uniquely named temp file in the destination
    directory, flushed and fsynced, then os.replace()'d into place (an atomic
    rename on POSIX). This eliminates the interleaved/truncated writes that occur
    when several parallel workers persist the cache at once. Failures here are
    non-fatal: the in-memory database is already available to the caller.
    """
    directory = os.path.dirname(pickle_file_path) or '.'
    try:
        fd, tmp_path = tempfile.mkstemp(prefix='_db.', suffix='.pkl.tmp', dir=directory)
    except OSError:
        return
    published = False
    try:
        with os.fdopen(fd, 'wb') as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, pickle_file_path)
        published = True
    except Exception:
        pass
    finally:
        if not published:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def load_airfoil_database_from_files(data_folder: str) -> list[Airfoil]:
    bundled = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATA_FOLDER)
    if data_folder == DATA_FOLDER and not os.path.isdir(data_folder) and os.path.isdir(bundled):
        data_folder = bundled
    pickle_file_path = os.path.join(data_folder, '_db.pkl')

    cached = _load_cached_db(pickle_file_path)
    if cached is not None:
        return cached

    airfoils_db, complete = _build_db_from_dat_files(data_folder)

    if airfoils_db and complete:
        _atomic_pickle_dump(airfoils_db, pickle_file_path)

    return airfoils_db

def get_baselines(data_folder: str, expected_names: list[str]) -> list[Airfoil]:
    """Loads exact specified baselines from the database."""
    global _CACHED_AIRFOIL_DB_DICT
    
    if _CACHED_AIRFOIL_DB_DICT is None:
        full_db = load_airfoil_database_from_files(data_folder)
        _CACHED_AIRFOIL_DB_DICT = {af.name: af for af in full_db}
        
    baselines = []
    for name in expected_names:
        if name in _CACHED_AIRFOIL_DB_DICT:
            baselines.append(_CACHED_AIRFOIL_DB_DICT[name])
        else:
            raise RuntimeError(f"Requested baseline '{name}' not found in the airfoil database.")
            
    return baselines
