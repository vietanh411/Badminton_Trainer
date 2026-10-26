"""Time-weighted court occupancy heatmaps.

Each position sample contributes the *seconds of observation it represents* (see
``sample_weights``), not a count of frames. A 60 fps clip and a 15 fps clip of the same
movement therefore give the same map, and tracking gaps contribute nothing.

The density is a weighted 2-D histogram on a metric grid convolved with a Gaussian kernel,
which is exactly a Gaussian kernel density estimate evaluated on the grid. Mass that the kernel
pushes outside the grid is renormalised back so totals stay exact.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from vision.geometry.court import COURT_LENGTH, COURT_WIDTH


@dataclass
class HeatmapConfig:
    cell_m: float = 0.10
    sigma_m: float = 0.30  # Gaussian kernel bandwidth; 0 = plain binning
    margin_m: float = 1.5  # grid extends beyond the doubles court (matches the trajectory bounds check)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Heatmap:
    seconds: np.ndarray  # (ny, nx) seconds of occupancy per cell
    x_edges: np.ndarray
    y_edges: np.ndarray
    total_s: float
    config: HeatmapConfig

    @property
    def share(self) -> np.ndarray:
        """Fraction of observed time per cell (sums to 1) — comparable across sessions."""
        return self.seconds / self.total_s if self.total_s > 0 else self.seconds

    @property
    def density_s_per_m2(self) -> np.ndarray:
        return self.seconds / (self.config.cell_m**2)

    def area_for_share(self, q: float = 0.9) -> float:
        """Smallest area (m²) containing a fraction ``q`` of the observed time."""
        if self.total_s <= 0:
            return 0.0
        v = np.sort(self.share.ravel())[::-1]
        k = int(np.searchsorted(np.cumsum(v), q)) + 1
        return k * self.config.cell_m**2

    def to_payload(self, normalize: str = "share", decimals: int = 6) -> dict:
        grid = {"share": self.share, "seconds": self.seconds, "density": self.density_s_per_m2}[normalize]
        return {
            "normalize": normalize,
            "units": {"share": "fraction of observed time", "seconds": "seconds per cell", "density": "seconds per m²"}[
                normalize
            ],
            "cell_m": self.config.cell_m,
            "sigma_m": self.config.sigma_m,
            "x0": float(self.x_edges[0]),
            "y0": float(self.y_edges[0]),
            "nx": int(len(self.x_edges) - 1),
            "ny": int(len(self.y_edges) - 1),
            "max": float(grid.max()) if grid.size else 0.0,
            "total_s": self.total_s,
            "area_50_m2": self.area_for_share(0.5),
            "area_90_m2": self.area_for_share(0.9),
            "values": np.round(grid, decimals).tolist(),
        }


def compute_heatmap(x: np.ndarray, y: np.ndarray, weights_s: np.ndarray, cfg: HeatmapConfig | None = None) -> Heatmap:
    cfg = cfg or HeatmapConfig()
    m = cfg.margin_m
    nx = int(round((COURT_WIDTH + 2 * m) / cfg.cell_m))
    ny = int(round((COURT_LENGTH + 2 * m) / cfg.cell_m))
    x_edges = -m + np.arange(nx + 1) * cfg.cell_m
    y_edges = -m + np.arange(ny + 1) * cfg.cell_m
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(weights_s, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    hist, _, _ = np.histogram2d(y[ok], x[ok], bins=[y_edges, x_edges], weights=w[ok])
    total = float(hist.sum())  # time inside the grid (positions beyond it were already rejected upstream)
    if cfg.sigma_m > 0 and total > 0:
        smoothed = gaussian_filter(hist, sigma=cfg.sigma_m / cfg.cell_m, mode="constant")
        s = smoothed.sum()
        hist = smoothed * (hist.sum() / s) if s > 0 else smoothed
    return Heatmap(seconds=hist, x_edges=x_edges, y_edges=y_edges, total_s=total, config=cfg)
