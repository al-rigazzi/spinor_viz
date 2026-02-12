"""
Spinor Visualization Package

Interactive visualization of spinors using the hyperchord representation.
"""

from spinor_viz.core import (
    Spinor,
    pauli,
    spinor_rotation,
    spinor_to_halfangles,
    halfangles_to_spinor,
    spinor_to_bloch_vector,
    fix_gauge,
)
from spinor_viz.visualization import (
    plot_spinor_2d,
    plot_spinor_3d,
    plot_complex,
)

__version__ = "0.1.0"

__all__ = [
    "Spinor",
    "pauli",
    "spinor_rotation",
    "spinor_to_halfangles",
    "halfangles_to_spinor",
    "spinor_to_bloch_vector",
    "fix_gauge",
    "plot_spinor_2d",
    "plot_spinor_3d",
    "plot_complex",
]
