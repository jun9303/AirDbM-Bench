import numpy as np

NUM_POINTS_INTERP: int = 161
THETA_HALF: np.ndarray = np.linspace(0.0, np.pi, NUM_POINTS_INTERP // 2 + 1)
X_INTERP_HALF_ASC: np.ndarray = 0.5 * (1.0 - np.cos(THETA_HALF))[1:]
X_INTERP_HALF_DESC: np.ndarray = (0.5 * (1.0 - np.cos(THETA_HALF)))[::-1]
X_INTERP: np.ndarray = np.concatenate((X_INTERP_HALF_DESC, X_INTERP_HALF_ASC))

DATA_FOLDER: str = 'airfoilDB'

EXPECTED_BASELINES: list[str] = [
    "E195  (11.82%)", "FX 79-W-660A", "GOE 531 AIRFOIL", "EPPLER 864 STRUT AIRFOIL",
    "RONCZ R1145MS MAIN ELEMENT", "CHEN AIRFOIL", "Griffith 30% Suction Airfoil", "S9104", 
    "AH 93-W-480B", "AH 81-K-144 W-F KLAPPE", "EPPLER 664 (EXTENDED) AIRFOIL", "SARATOV AIRFOIL"
]

MIN_INTERIOR_THICKNESS: float = 1e-3  # Chord-normalized interior thickness floor for geometry correction
MIN_TRAILING_EDGE_THICKNESS: float = 1e-4  # Keep upper/lower TE endpoints from crossing

XFOIL_APP: str = 'bin/xfoil-ubuntu22.sif'  # Default apptainer image path for portable XFOIL execution
REYNOLDS: float = 1e6  # Default chord-based Reynolds number Re_c for XFOIL simulations
MACH: float = 0.0 # Default Mach number for XFOIL simulations (incompressible flow)
N_CRIT: float = 9.0 # Default N_crit for transition prediction in XFOIL
CD_BRANCH_TOL: float = 0.15       # max Cd disagreement between coarse and refined at a shared alpha
CD_DROP_TOL: float = 0.10         # max benign Cd fall AFTER the drag bucket (calibrated: 4.6% worst)
CD_BUCKET_DROP_TOL: float = 0.35  # max benign Cd fall WHILE still descending into the bucket
COARSE_ALPHA_STEP: float = 0.5 # Step size for initial coarse alpha scan in XFOIL
REFINED_ALPHA_STEP: float = 0.1 # Step size for refined alpha scan around identified centers in XFOIL
REFINE_ALPHA_WINDOW: float = 1.0 # Window size around identified centers for refined scanning in XFOIL

LAMINAR_XTR = 0.95   # a transition location at or beyond this is "no transition before the TE"
