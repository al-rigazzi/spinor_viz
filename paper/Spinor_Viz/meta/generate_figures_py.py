"""Generate paper figures using the Python plotting style.

Outputs are written alongside existing assets with a _py suffix.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from spinor_viz.core import Spinor
from spinor_viz.visualization import (
    plot_planar_chord_panel,
    plot_spinor_2d,
    plot_spinor_3d,
)


matplotlib.use("Agg")

FIG_DIR = Path(__file__).resolve().parents[1] / "viz_figures"
VIEW_ANGLES_3D = (25, 15)

def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _clean_2d_axes(ax: plt.Axes) -> None:
    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])


def _set_paper_fonts() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{mathptmx}",
        "font.size": 11,
        "axes.titlesize": 11,
        "axes.labelsize": 11,
    })


def generate_planar_chords() -> None:
    """Generate the planar chord construction panels."""
    _set_paper_fonts()

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_planar_chord_panel(np.deg2rad(50.0), ax=ax, label_angle=r"$\vartheta$", red_negative=False)
    _save_figure(fig, FIG_DIR / "planar_chords_theta_py.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    plot_planar_chord_panel(np.deg2rad(50.0) + np.pi, ax=ax, label_angle=r"$\vartheta+\pi$", red_negative=True)
    _save_figure(fig, FIG_DIR / "planar_chords_theta_pi_py.png")


def generate_hyperchords_3d() -> None:
    """Generate the 3D hyperchords schematic."""
    spinor = Spinor.from_angles(theta=np.pi / 3, phi=np.pi / 4)
    ax = plot_spinor_3d(spinor, view_angles=VIEW_ANGLES_3D, show_legend=False)
    fig = ax.figure
    _save_figure(fig, FIG_DIR / "hyperchords_3d_py.png")


def generate_spinor_2d() -> None:
    """Generate the 2D circle and ellipse projections."""
    spinor = Spinor.from_angles(theta=np.pi / 3, phi=np.pi / 4)

    ax = plot_spinor_2d(spinor, show_ellipses=False, show_circles=True, figsize=(6, 6))
    angles = np.linspace(0, 2 * np.pi, 100)
    radius = spinor.norm ** 2
    _clean_2d_axes(ax)
    fig = ax.figure
    _save_figure(fig, FIG_DIR / "spinor2d_circles_py.png")

    ax = plot_spinor_2d(spinor, show_ellipses=True, show_circles=False, figsize=(6, 6))
    _clean_2d_axes(ax)
    fig = ax.figure
    _save_figure(fig, FIG_DIR / "spinor2d_ellipses_py.png")


def _rotation_frames() -> list[float]:
    return [k * (np.pi / 2) for k in range(8)]


def generate_anim_3d() -> None:
    """Generate the 3D z-rotation animation frames."""
    spinor = Spinor.from_angles(theta=np.pi / 3, phi=np.pi / 4)

    for idx, omega in enumerate(_rotation_frames(), start=1):
        rotated = spinor.rotate(np.array([0.0, 0.0, omega]))
        ax = plot_spinor_3d(rotated, view_angles=VIEW_ANGLES_3D, show_legend=False)
        fig = ax.figure
        _save_figure(fig, FIG_DIR / f"anim_z_{idx:02d}_py.png")


def generate_anim_2d() -> None:
    """Generate the 2D z-rotation animation frames."""
    spinor = Spinor.from_angles(theta=np.pi / 3, phi=np.pi / 4)
    angles = np.linspace(0, 2 * np.pi, 100)
    radius = spinor.norm ** 2

    for idx, omega in enumerate(_rotation_frames(), start=1):
        rotated = spinor.rotate(np.array([0.0, 0.0, omega]))

        ax = plot_spinor_2d(rotated, show_ellipses=True, show_circles=False, figsize=(6, 6))
        _clean_2d_axes(ax)
        fig = ax.figure
        _save_figure(fig, FIG_DIR / f"anim2d_ellipse_{idx:02d}_py.png")

        ax = plot_spinor_2d(rotated, show_ellipses=False, show_circles=True, figsize=(6, 6))
        # ax.plot(radius * np.cos(angles), radius * np.sin(angles), "k", linewidth=1.5)
        _clean_2d_axes(ax)
        fig = ax.figure
        _save_figure(fig, FIG_DIR / f"anim2d_circle_{idx:02d}_py.png")


def main() -> None:
    generate_planar_chords()
    generate_hyperchords_3d()
    generate_spinor_2d()
    generate_anim_3d()
    generate_anim_2d()


if __name__ == "__main__":
    main()
