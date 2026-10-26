"""Matplotlib renderings of the top-down court (heatmap / trajectory exports and the PDF report)."""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from vision.analytics.heatmap import Heatmap  # noqa: E402
from vision.geometry.court import COURT_LENGTH, COURT_WIDTH, court_lines  # noqa: E402

BG = "#0b1220"
COURT = "#0f1a2e"
LINE = "#c9d4e5"
CYAN = "#22d3ee"
LIME = "#a3e635"

HEAT_CMAP = LinearSegmentedColormap.from_list(
    "bv_heat",
    [
        (0.0, (0.06, 0.10, 0.18, 0.0)),
        (0.15, (0.05, 0.35, 0.55, 0.55)),
        (0.45, (0.13, 0.83, 0.93, 0.8)),
        (0.75, (0.64, 0.90, 0.21, 0.9)),
        (1.0, (0.99, 0.95, 0.55, 1.0)),
    ],
)


def draw_court(ax, *, color: str = LINE, lw: float = 1.2, background: bool = True) -> None:
    if background:
        ax.add_patch(plt.Rectangle((0, 0), COURT_WIDTH, COURT_LENGTH, color=COURT, zorder=0))
    for ln in court_lines():
        style = "--" if ln.kind == "net" else "-"
        ax.plot(
            [ln.start[0], ln.end[0]],
            [ln.start[1], ln.end[1]],
            style,
            color=color,
            lw=lw * (1.4 if ln.kind == "net" else 1),
            zorder=3,
            solid_capstyle="butt",
        )
    ax.set_xlim(-1.0, COURT_WIDTH + 1.0)
    ax.set_ylim(-1.0, COURT_LENGTH + 1.0)
    ax.set_aspect("equal")
    ax.axis("off")


def heatmap_figure(
    hm: Heatmap,
    *,
    title: str | None = None,
    points: np.ndarray | None = None,
    trajectory: np.ndarray | None = None,
    opacity: float = 0.9,
    normalize: str = "share",
):
    fig, ax = plt.subplots(figsize=(4.2, 8.0), dpi=150)
    fig.patch.set_facecolor(BG)
    draw_court(ax)
    grid = {"share": hm.share * 100, "seconds": hm.seconds, "density": hm.density_s_per_m2}[normalize]
    label = {"share": "% of observed time per cell", "seconds": "seconds per cell", "density": "seconds per m²"}[
        normalize
    ]
    if hm.total_s > 0:
        im = ax.imshow(
            grid,
            origin="lower",
            extent=(hm.x_edges[0], hm.x_edges[-1], hm.y_edges[0], hm.y_edges[-1]),
            cmap=HEAT_CMAP,
            alpha=opacity,
            zorder=2,
            interpolation="bilinear",
            vmin=0,
        )
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(label, color=LINE, fontsize=7)
        cb.ax.tick_params(colors=LINE, labelsize=6)
    if trajectory is not None and len(trajectory):
        ax.plot(trajectory[:, 0], trajectory[:, 1], color=CYAN, lw=0.6, alpha=0.6, zorder=4)
    if points is not None and len(points):
        ax.scatter(points[:, 0], points[:, 1], s=1.5, color=LIME, alpha=0.5, zorder=5, linewidths=0)
    if title:
        ax.set_title(title, color=LINE, fontsize=9)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return path


def figure_bytes(fig, fmt: str = "png") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
