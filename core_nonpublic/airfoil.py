import numpy as np
import matplotlib.pyplot as plt
from .constants import X_INTERP

class Airfoil:
    def __init__(self, airfoil_id: int, name: str, colloc_vec: np.ndarray, x_raw: np.ndarray, y_raw: np.ndarray):
        self.airfoil_id = airfoil_id
        self.name = name
        self.colloc_vec = colloc_vec
        self.x_raw = x_raw
        self.y_raw = y_raw
        self.xfoil_result = None

    def __str__(self) -> str:
        return f"Airfoil(ID={self.airfoil_id}, Name='{self.name}', Raw Data Points={len(self.x_raw)})"

    def get_raw_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self.x_raw, self.y_raw

    def get_interpolated_data(self) -> np.ndarray:
        return self.colloc_vec

    def plot(self, save_path: str | None = None) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_raw, self.y_raw, 'o', label='Raw Data', alpha=0.7, markersize=4)
        plt.plot(np.append(X_INTERP, X_INTERP[0]),
                 np.append(self.colloc_vec, self.colloc_vec[0]),
                 '-', label='Interpolated Data', linewidth=1.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"Airfoil: {self.name}", fontsize=14)
        plt.xlabel("x", fontsize=12)
        plt.ylabel("y", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
