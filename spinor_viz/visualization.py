"""
Visualization utilities for spinors.

This module provides 2D and 3D plotting functions for visualizing spinors
using a 2D dihedral to 3D plane-dihedral construction, with both static matplotlib plots
and interactive Plotly visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Polygon
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
import plotly.graph_objects as go
from typing import Optional, Tuple, Union, Dict, Any

from spinor_viz.core import (
    Spinor,
    spinor_to_halfangles,
    spinor_to_bloch_vector,
    normalize,
    rotate_vector_around_axis,
)

# Colorblind-friendly palette (Wong palette)
# https://www.nature.com/articles/nmeth.1618
COLOR_UP = '#009E73'      # Teal/bluish-green for c_up
COLOR_DOWN = '#E69F00'    # Orange for c_down
COLOR_REAL = '#0072B2'    # Blue for real part arrows
COLOR_IMAG = '#D55E00'    # Vermillion for imaginary part arrows
COLOR_BLOCH = '#CC79A7'   # Reddish-purple for Bloch vector

# Line widths
LINEWIDTH_UP = 3.0        # Thicker for c_up
LINEWIDTH_DOWN = 1.5      # Thinner for c_down


def _add_static_legend_3d(ax: 'Axes3D', show_rotation_axis: bool = False) -> None:
    """
    Add static legend entries for all plottable entities in 3D spinor plots.

    This creates invisible dummy plot elements with labels so the legend
    always shows all possible items regardless of what's actually displayed.

    Parameters
    ----------
    ax : Axes3D
        The 3D axes to add legend entries to.
    show_rotation_axis : bool
        Whether to include the rotation axis entry.
    """
    # Dummy invisible points at origin for legend entries
    dummy = np.array([[0, 0, 0]])

    # Equator
    ax.plot([], [], [], 'k', linewidth=1.5, label='Equator')

    # Rotation axis (only if requested)
    if show_rotation_axis:
        ax.plot([], [], [], 'g:', linewidth=2, label='Rotation axis')

    # Key points
    ax.scatter([], [], [], color='black', s=50, label='Shared point')
    ax.scatter([], [], [], color=COLOR_UP, s=50, label='Upper equator point', marker="^")
    ax.scatter([], [], [], color=COLOR_DOWN, s=50, label='Lower equator point', marker="v")

    # Plane-dihedral faces
    ax.plot([], [], [], color=COLOR_UP, linewidth=LINEWIDTH_UP, label='Upper plane-dihedral face (↑)')
    ax.plot([], [], [], color=COLOR_DOWN, linewidth=LINEWIDTH_DOWN, label='Lower plane-dihedral face (↓)')

    # Diameter
    ax.plot([], [], [], 'k', linewidth=1, label='Diameter')

    # Bloch vector
    ax.plot([], [], [], color=COLOR_BLOCH, linewidth=2, label='Bloch vector')


def _is_displayable(v: np.ndarray, tol: float = 1e-8) -> bool:
    """Check if a vector has significant magnitude for display."""
    return np.linalg.norm(v) > tol


def _generate_circle_points(
    center: np.ndarray,
    radius: float,
    normal: np.ndarray,
    n_points: int = 50
) -> np.ndarray:
    """
    Generate points on a circle in 3D space.

    Parameters
    ----------
    center : np.ndarray
        Center of the circle (3D point).
    radius : float
        Radius of the circle.
    normal : np.ndarray
        Normal vector to the plane of the circle.
    n_points : int
        Number of points to generate.

    Returns
    -------
    np.ndarray
        Array of shape (n_points, 3) with circle points.
    """
    normal = normalize(normal)

    # Find two orthogonal vectors in the plane
    if abs(normal[2]) < 0.9:
        u = normalize(np.cross(normal, [0, 0, 1]))
    else:
        u = normalize(np.cross(normal, [1, 0, 0]))
    v = normalize(np.cross(normal, u))

    angles = np.linspace(0, 2 * np.pi, n_points)
    points = center + radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v))
    return points


def _inclined_face_3d(
    d1: np.ndarray,
    d2: np.ndarray,
    rot_angle: float,
    is_up: bool = False,
    n_points: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the inclined circle section (plane-dihedral face) in 3D.

    Parameters
    ----------
    d1 : np.ndarray
        First diameter point (on equator).
    d2 : np.ndarray
        Second diameter point (shared point).
    rot_angle : float
        Rotation angle for the arrow position.
    is_up : bool
        Whether this is the upper (True) or lower (False) plane-dihedral face.
    n_points : int
        Number of points for the circle.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (circle_points, diameter_endpoints, arrow_start_point)
    """
    center = (d2 + d1) / 2
    center_d2 = d2 - center
    radius = np.linalg.norm(center_d2)

    if radius < 1e-10:
        # Degenerate case
        angles = np.linspace(0, 2 * np.pi, n_points)
        circle = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n_points)])
        return circle, np.array([[-1, 0, 0], [1, 0, 0]]), np.array([1, 0, 0])

    center_d2 = center_d2 / radius

    # Orthogonal vector in the xy-plane
    ortho_vec = normalize(np.array([center_d2[1], -center_d2[0], 0]))

    # Rotation angle to tilt the circle
    rot_degrees = np.degrees(np.arcsin(d2[2] / (radius * 2))) if radius > 1e-10 else 0

    # Generate circle in xy-plane and rotate
    angles = np.linspace(0, 2 * np.pi, n_points)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n_points)])
    circle_pts = rotate_vector_around_axis(circle_pts, ortho_vec, rot_degrees)
    circle_pts = circle_pts * radius + center

    # Diameter endpoints
    x_dir_pts = np.array([[-1, 0, 0], [1, 0, 0]], dtype=float)
    x_dir_pts = rotate_vector_around_axis(x_dir_pts, ortho_vec, rot_degrees)
    x_dir_pts = x_dir_pts * radius + center

    # Arrow start position
    sign = -1 if is_up else 1
    arrow_start = sign * np.array([np.cos(rot_angle), np.sin(rot_angle), 0])
    arrow_start = rotate_vector_around_axis(arrow_start, ortho_vec, rot_degrees)
    arrow_start = arrow_start * radius + center

    return circle_pts, x_dir_pts, arrow_start


def _meridian_points(up_and_eq: np.ndarray, n_points: int = 50) -> np.ndarray:
    """Generate points along a meridian passing through a point on the equator."""
    radius = np.linalg.norm(up_and_eq)
    if radius < 1e-10:
        return np.zeros((n_points, 3))

    vec = up_and_eq / radius
    ortho_vec = np.array([-vec[1], vec[0], 0])

    rot_degrees = np.linspace(0, 360, n_points)
    meridian_pts = rotate_vector_around_axis(
        np.tile(vec, (n_points, 1)),
        ortho_vec,
        rot_degrees
    ) * radius

    return meridian_pts


def plot_spinor_3d(
    spinor: Union[Spinor, np.ndarray],
    ax: Optional[Axes3D] = None,
    view_angles: Tuple[float, float] = (30, 45),
    show_rotation_axis: Optional[np.ndarray] = None,
    figsize: Tuple[int, int] = (10, 10),
    show_legend: bool = True,
    show_axis_arrows: bool = True,
    **kwargs
) -> Axes3D:
    """
    Create a 3D visualization of a spinor using the plane-dihedral construction.

    The visualization shows:
    - Equator circle (black)
    - Upper plane-dihedral face (circle section, blue)
    - Lower plane-dihedral face (circle section, red)
    - Bloch vector (arrow from origin)
    - Arrows indicating spinor component values

    Parameters
    ----------
    spinor : Spinor or np.ndarray
        The spinor to visualize.
    ax : Axes3D, optional
        Existing 3D axes to plot on. If None, creates new figure.
    view_angles : Tuple[float, float]
        (elevation, azimuth) viewing angles in degrees.
    show_rotation_axis : np.ndarray, optional
        If provided, shows this rotation axis as a green dotted line.
    figsize : Tuple[int, int]
        Figure size if creating new figure.
    show_legend : bool
        If True (default), shows the legend in the lower right corner.
    show_axis_arrows : bool
        If True, draws coordinate axes as arrows with arrowheads at +1.
    **kwargs
        Additional keyword arguments for customization.

    Returns
    -------
    Axes3D
        The matplotlib 3D axes with the plot.

    Examples
    --------
    >>> from spinor_viz import Spinor, plot_spinor_3d
    >>> s = Spinor.from_angles(np.pi/3, np.pi/4)
    >>> plot_spinor_3d(s)
    """
    if isinstance(spinor, Spinor):
        s = spinor.components
    else:
        s = np.asarray(spinor).flatten()

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    # Extract spinor parameters
    half_theta, half_alpha, half_phi = spinor_to_halfangles(s)

    phi = half_phi * 2
    alpha = half_alpha * 2
    sp_length = np.linalg.norm(s)
    sphere_radius = sp_length ** 2
    s_up = s[0] * sp_length * 2
    s_down = s[1] * sp_length * 2

    plot_up = _is_displayable(np.array([s_up]))
    plot_down = _is_displayable(np.array([s_down]))

    up_eq_angle = phi
    down_eq_angle = phi + np.pi

    # Osculating points
    up_and_equator = sphere_radius * np.array([np.cos(up_eq_angle), np.sin(up_eq_angle), 0])
    down_and_equator = sphere_radius * np.array([np.cos(down_eq_angle), np.sin(down_eq_angle), 0])
    up_and_down = up_and_equator + np.abs(s_up) * np.array([
        np.cos(half_theta) * np.cos(down_eq_angle),
        np.cos(half_theta) * np.sin(down_eq_angle),
        np.sin(half_theta)
    ])

    # Equator circle
    n_pts = 50
    angles = np.linspace(0, 2 * np.pi, n_pts)
    circle_pts = sphere_radius * np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n_pts)])
    ax.plot(circle_pts[:, 0], circle_pts[:, 1], circle_pts[:, 2], 'k', linewidth=1.5)

    # Show rotation axis if provided
    if show_rotation_axis is not None:
        rot_axis = normalize(show_rotation_axis)
        factor = sp_length * 1.1 / 2
        ax.plot(
            [0, factor * rot_axis[0]],
            [0, factor * rot_axis[1]],
            [0, factor * rot_axis[2]],
            'g:', linewidth=2
        )

    if plot_up and plot_down:
        # Key points
        ax.scatter(*up_and_down, color='black', s=50, zorder=5)
        ax.scatter(*up_and_equator, color=COLOR_UP, s=100, zorder=5, marker="^")
        ax.scatter(*down_and_equator, color=COLOR_DOWN, s=100, zorder=5, marker="v")

        # Meridian
        meridian_pts = _meridian_points(up_and_equator)
        meridian_plot, = ax.plot(meridian_pts[:, 0], meridian_pts[:, 1], meridian_pts[:, 2],
                'c', linewidth=1)
        meridian_plot.set_dashes([1,5])

    # Upper plane-dihedral face
    if plot_up:
        up_circle, up_x, rot_pt_up = _inclined_face_3d(
            up_and_equator, up_and_down, alpha + phi, is_up=True
        )
        ax.plot(up_circle[:, 0], up_circle[:, 1], up_circle[:, 2], color=COLOR_UP, linewidth=LINEWIDTH_UP)
        ax.plot(up_x[:, 0], up_x[:, 1], up_x[:, 2], 'k', linewidth=1)

        # Draw arrows (real with arrowhead, imaginary without)
        if _is_displayable(up_x[1] - rot_pt_up):
            _draw_arrow_3d(ax, rot_pt_up, up_x[1], COLOR_REAL, np.real(s_up), is_real=True)
        if _is_displayable(up_x[0] - rot_pt_up):
            _draw_arrow_3d(ax, rot_pt_up, up_x[0], COLOR_IMAG, np.imag(s_up), is_real=False)

    # Lower plane-dihedral face
    if plot_down:
        down_circle, down_x, rot_pt_down = _inclined_face_3d(
            down_and_equator, up_and_down, -(alpha - phi), is_up=False
        )
        ax.plot(down_circle[:, 0], down_circle[:, 1], down_circle[:, 2], color=COLOR_DOWN, linewidth=LINEWIDTH_DOWN)
        ax.plot(down_x[:, 0], down_x[:, 1], down_x[:, 2], 'k', linewidth=1)  # No label to avoid duplicate

        # Draw arrows (real with arrowhead, imaginary without)
        if _is_displayable(rot_pt_down - down_x[0]):
            _draw_arrow_3d(ax, rot_pt_down, down_x[0], COLOR_REAL, np.real(s_down), is_real=True)
        if _is_displayable(rot_pt_down - down_x[1]):
            _draw_arrow_3d(ax, rot_pt_down, down_x[1], COLOR_IMAG, np.imag(s_down), is_real=False)

    # Bloch vector
    vec = np.real(spinor_to_bloch_vector(s))
    ax.quiver(0, 0, 0, vec[1], vec[2], vec[3], color=COLOR_BLOCH, arrow_length_ratio=0.1, linewidth=2)

    # Formatting - axes from -1 to 1, no ticks
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])  # Only show Z >= 0
    ax.set_xlabel('X', labelpad=-12)
    ax.set_ylabel('Y', labelpad=-12)
    ax.set_zlabel('Z', labelpad=-17)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    # Ensure axis labels remain visible
    ax.xaxis.label.set_visible(True)
    ax.yaxis.label.set_visible(True)
    ax.zaxis.label.set_visible(True)
    ax.view_init(elev=view_angles[0], azim=view_angles[1])
    ax.set_box_aspect([1, 1, 0.5])  # Adjust aspect for half Z range

    # Remove grids on planes but keep axis lines
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('none')
    ax.yaxis.pane.set_edgecolor('none')
    ax.zaxis.pane.set_edgecolor('none')
    ax.grid(False)

    # Draw axis arrows if requested
    if show_axis_arrows:
        # Hide default axes completely
        ax.set_axis_off()

        arrow_color = 'gray'
        arrow_alpha = 0.7
        # X axis: from -1 to 1, starting at (-1, -1, 0)
        ax.quiver(-1, -1, 0, 2, 0, 0, color=arrow_color, alpha=arrow_alpha,
                  arrow_length_ratio=0.05, linewidth=1.5)
        ax.text(1, -1, 0.15, 'X', color=arrow_color, fontsize=10, ha='center', va='top')
        # Y axis: from -1 to 1, starting at (-1, -1, 0)
        ax.quiver(-1, -1, 0, 0, 2, 0, color=arrow_color, alpha=arrow_alpha,
                  arrow_length_ratio=0.05, linewidth=1.5)
        ax.text(-1, 1, 0.15, 'Y', color=arrow_color, fontsize=10, ha='center', va='top')
        # Z axis: from 0 to 1, starting at (-1, -1, 0)
        ax.quiver(-1, -1, 0, 0, 0, 1, color=arrow_color, alpha=arrow_alpha,
                  arrow_length_ratio=0.1, linewidth=1.5)
        ax.text(-1, -1, 1.15, 'Z', color=arrow_color, fontsize=10, ha='center', va='top')

    # Add static legend entries and show legend outside plot (to the right)
    if show_legend:
        _add_static_legend_3d(ax, show_rotation_axis=show_rotation_axis is not None)
        ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), framealpha=0.9)

    return ax


def _draw_arrow_3d(
    ax: Axes3D,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    magnitude: float,
    is_real: bool = True
) -> None:
    """Draw a 3D arrow, with style depending on magnitude sign.

    Real part arrows get arrowheads, imaginary part arrows do not.
    Positive values: solid line with arrowhead; Negative values: dashed line with filled circle.
    """
    if magnitude < 0:
        linestyle = ':'
        start, end = end, start
        end_marker = 'o'  # Filled circle for negative
    else:
        linestyle = '-'
        end_marker = '>'  # Arrowhead for positive

    ax.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]],
            color=color, linestyle=linestyle, linewidth=2)

    # Add end marker for real part only
    if is_real and abs(magnitude) > 1e-8:
        direction = end - start
        length = np.linalg.norm(direction)
        if length > 1e-10:
            # Draw marker at the end: arrowhead for positive, filled circle for negative
            ax.scatter(*end, color=color, s=40, marker=end_marker, zorder=5)


def plot_spinor_2d(
    spinor: Union[Spinor, np.ndarray],
    ax: Optional[plt.Axes] = None,
    show_ellipses: bool = True,
    show_circles: bool = False,
    figsize: Tuple[int, int] = (8, 8),
    **kwargs
) -> plt.Axes:
    """
    Create a 2D projection visualization of a spinor.

    Projects the 3D plane-dihedral faces onto the xy-plane.

    Parameters
    ----------
    spinor : Spinor or np.ndarray
        The spinor to visualize.
    ax : plt.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    show_ellipses : bool
        If True, shows the projected ellipses (default).
    show_circles : bool
        If True, shows undistorted circles (2D representation).
    figsize : Tuple[int, int]
        Figure size if creating new figure.
    **kwargs
        Additional customization options.

    Returns
    -------
    plt.Axes
        The matplotlib axes with the plot.
    """
    if isinstance(spinor, Spinor):
        s = spinor.components
    else:
        s = np.asarray(spinor).flatten()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Extract spinor parameters
    half_theta, half_alpha, half_phi = spinor_to_halfangles(s)

    phi = half_phi * 2
    alpha = half_alpha * 2
    sp_length = np.linalg.norm(s)
    sphere_radius = sp_length ** 2
    s_up = s[0] * sp_length * 2
    s_down = s[1] * sp_length * 2

    plot_up = _is_displayable(np.array([s_up]))
    plot_down = _is_displayable(np.array([s_down]))

    up_eq_angle = phi
    down_eq_angle = phi + np.pi

    # 3D points for projection
    up_and_equator_3d = sphere_radius * np.array([np.cos(up_eq_angle), np.sin(up_eq_angle), 0])
    down_and_equator_3d = sphere_radius * np.array([np.cos(down_eq_angle), np.sin(down_eq_angle), 0])
    up_and_down_3d = up_and_equator_3d + np.abs(s_up) * np.array([
        np.cos(half_theta) * np.cos(down_eq_angle),
        np.cos(half_theta) * np.sin(down_eq_angle),
        np.sin(half_theta)
    ])

    # 2D projections
    up_and_equator = up_and_equator_3d[:2]
    down_and_equator = down_and_equator_3d[:2]
    up_and_down = up_and_down_3d[:2]

    # Equator circle
    n_pts = 50
    angles = np.linspace(0, 2 * np.pi, n_pts)

    if show_ellipses:
        circle_pts = sphere_radius * np.column_stack([np.cos(angles), np.sin(angles)])
        ax.plot(circle_pts[:, 0], circle_pts[:, 1], 'k', linewidth=1.5)

        if plot_up and plot_down:
            ax.scatter(*up_and_down, color='black', s=25, zorder=5)
            ax.scatter(*up_and_equator, color=COLOR_UP, s=50, zorder=5, marker='^')
            ax.scatter(*down_and_equator, color=COLOR_DOWN, s=50, zorder=5, marker='v')

            # Meridian projection
            meridian_pts = _meridian_points(up_and_equator_3d)
            ax.plot(meridian_pts[:, 0], meridian_pts[:, 1], 'c:', linewidth=0.1)

        # Upper plane-dihedral face
        if plot_up:
            up_circle_3d, up_x_3d, rot_pt_up_3d = _inclined_face_3d(
                up_and_equator_3d, up_and_down_3d, alpha + phi, is_up=True
            )
            ax.plot(up_circle_3d[:, 0], up_circle_3d[:, 1], color=COLOR_UP, linewidth=LINEWIDTH_UP)
            ax.plot(up_x_3d[:, 0], up_x_3d[:, 1], 'k', linewidth=1)

            rot_pt_up = rot_pt_up_3d[:2]
            up_x = up_x_3d[:, :2]
            if _is_displayable(up_x[1] - rot_pt_up):
                _draw_arrow_2d(ax, rot_pt_up, up_x[1], COLOR_REAL, np.real(s_up), is_real=True)
            if _is_displayable(up_x[0] - rot_pt_up):
                _draw_arrow_2d(ax, rot_pt_up, up_x[0], COLOR_IMAG, np.imag(s_up), is_real=False)

        # Lower plane-dihedral face
        if plot_down:
            down_circle_3d, down_x_3d, rot_pt_down_3d = _inclined_face_3d(
                down_and_equator_3d, up_and_down_3d, -(alpha - phi), is_up=False
            )
            ax.plot(down_circle_3d[:, 0], down_circle_3d[:, 1], color=COLOR_DOWN, linewidth=LINEWIDTH_DOWN)
            ax.plot(down_x_3d[:, 0], down_x_3d[:, 1], 'k', linewidth=1)

            rot_pt_down = rot_pt_down_3d[:2]
            down_x = down_x_3d[:, :2]
            if _is_displayable(rot_pt_down - down_x[0]):
                _draw_arrow_2d(ax, rot_pt_down, down_x[0], COLOR_REAL, np.real(s_down), is_real=True)
            if _is_displayable(rot_pt_down - down_x[1]):
                _draw_arrow_2d(ax, rot_pt_down, down_x[1], COLOR_IMAG, np.imag(s_down), is_real=False)

    if show_circles:
        # Alternative representation with undistorted circles
        if plot_up:
            up_circle_3d, up_x_3d, rot_pt_up_3d = _inclined_face_3d(
                up_and_equator_3d, up_and_down_3d, alpha + phi, is_up=True
            )
            center_3d = (up_and_down_3d + up_and_equator_3d) / 2
            radius_up = np.linalg.norm(up_and_down_3d - center_3d)

            dir_up = up_and_equator - up_and_down
            if np.linalg.norm(dir_up) > 1e-10:
                dir_up = dir_up / np.linalg.norm(dir_up)
                center_up = up_and_down + dir_up * radius_up
                u2d = dir_up
                v2d = np.array([-u2d[1], u2d[0]])

                u3d = up_and_equator_3d - center_3d
                u3d_norm = np.linalg.norm(u3d)
                if u3d_norm > 1e-10:
                    u3d = u3d / u3d_norm
                    normal = np.cross(up_circle_3d[1] - center_3d, up_circle_3d[2] - center_3d)
                    normal_norm = np.linalg.norm(normal)
                    if normal_norm > 1e-10:
                        normal = normal / normal_norm
                        v3d = np.cross(normal, u3d)
                        v3d_norm = np.linalg.norm(v3d)
                        if v3d_norm > 1e-10:
                            v3d = v3d / v3d_norm

                            def map_up_point(p3d: np.ndarray) -> np.ndarray:
                                offset = p3d - center_3d
                                x = np.dot(offset, u3d)
                                y = np.dot(offset, v3d)
                                angle = np.arctan2(y, x)
                                return center_up + radius_up * (np.cos(angle) * u2d + np.sin(angle) * v2d)

                circle_up = center_up + radius_up * np.column_stack([np.cos(angles), np.sin(angles)])
                ax.plot(circle_up[:, 0], circle_up[:, 1], color=COLOR_UP, linewidth=LINEWIDTH_UP)

                up_dot = center_up + dir_up * radius_up
                ax.scatter(*up_dot, color=COLOR_UP, s=50, zorder=5, marker='^')

                if "map_up_point" in locals():
                    rot_pt_up = map_up_point(rot_pt_up_3d)
                    up_x0 = map_up_point(up_x_3d[0])
                    up_x1 = map_up_point(up_x_3d[1])
                    if _is_displayable(up_x1 - rot_pt_up):
                        _draw_arrow_2d(ax, rot_pt_up, up_x1, COLOR_REAL, np.real(s_up), is_real=True)
                    if _is_displayable(up_x0 - rot_pt_up):
                        _draw_arrow_2d(ax, rot_pt_up, up_x0, COLOR_IMAG, np.imag(s_up), is_real=False)

        if plot_down:
            down_circle_3d, down_x_3d, rot_pt_down_3d = _inclined_face_3d(
                down_and_equator_3d, up_and_down_3d, -(alpha - phi), is_up=False
            )
            center_3d = (up_and_down_3d + down_and_equator_3d) / 2
            radius_down = np.linalg.norm(up_and_down_3d - center_3d)

            dir_down = down_and_equator - up_and_down
            if np.linalg.norm(dir_down) > 1e-10:
                dir_down = dir_down / np.linalg.norm(dir_down)
                center_down = up_and_down + dir_down * radius_down
                u2d = dir_down
                v2d = np.array([-u2d[1], u2d[0]])

                u3d = down_and_equator_3d - center_3d
                u3d_norm = np.linalg.norm(u3d)
                if u3d_norm > 1e-10:
                    u3d = u3d / u3d_norm
                    normal = np.cross(down_circle_3d[1] - center_3d, down_circle_3d[2] - center_3d)
                    normal_norm = np.linalg.norm(normal)
                    if normal_norm > 1e-10:
                        normal = normal / normal_norm
                        v3d = np.cross(normal, u3d)
                        v3d_norm = np.linalg.norm(v3d)
                        if v3d_norm > 1e-10:
                            v3d = v3d / v3d_norm

                            def map_down_point(p3d: np.ndarray) -> np.ndarray:
                                offset = p3d - center_3d
                                x = np.dot(offset, u3d)
                                y = np.dot(offset, v3d)
                                angle = np.arctan2(y, x)
                                return center_down + radius_down * (np.cos(angle) * u2d + np.sin(angle) * v2d)

                circle_down = center_down + radius_down * np.column_stack([np.cos(angles), np.sin(angles)])
                ax.plot(circle_down[:, 0], circle_down[:, 1], color=COLOR_DOWN, linewidth=LINEWIDTH_DOWN)

                down_dot = center_down + dir_down * radius_down
                ax.scatter(*down_dot, color=COLOR_DOWN, s=50, zorder=5, marker='v')

                if "map_down_point" in locals():
                    rot_pt_down = map_down_point(rot_pt_down_3d)
                    down_x0 = map_down_point(down_x_3d[0])
                    down_x1 = map_down_point(down_x_3d[1])
                    if _is_displayable(rot_pt_down - down_x0):
                        _draw_arrow_2d(ax, rot_pt_down, down_x0, COLOR_REAL, np.real(s_down), is_real=True)
                    if _is_displayable(rot_pt_down - down_x1):
                        _draw_arrow_2d(ax, rot_pt_down, down_x1, COLOR_IMAG, np.imag(s_down), is_real=False)

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim([-sphere_radius * 1.3, sphere_radius * 1.3])
    ax.set_ylim([-sphere_radius * 1.3, sphere_radius * 1.3])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Dihedral / 3D Plane-Dihedral Projection')

    return ax


def _draw_arrow_2d(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    magnitude: float,
    is_real: bool = True
) -> None:
    """Draw a 2D arrow with appropriate style based on magnitude.

    Real part arrows get arrowheads, imaginary part arrows do not.
    Positive values: solid line with arrowhead; Negative values: dashed line with filled circle.
    """
    if abs(magnitude) < 1e-8:
        return

    if magnitude < 0:
        linestyle = '--'
        start, end = end, start
        draw_end_circle = True
    else:
        linestyle = '-'
        draw_end_circle = False

    direction = end - start
    length = np.linalg.norm(direction)

    if length < 1e-10:
        return

    ax.plot([start[0], end[0]], [start[1], end[1]],
            color=color, linestyle=linestyle, linewidth=2)

    # Only draw end marker for real part
    if is_real:
        if draw_end_circle:
            # Filled circle for negative values
            circle_radius = length * 0.04
            circle = Circle(end, circle_radius, facecolor=color, edgecolor=color)
            ax.add_patch(circle)
        else:
            # Arrowhead for positive values
            dir_norm = direction / length
            perp = np.array([-dir_norm[1], dir_norm[0]])
            head_size = length * 0.04

            head1 = end - dir_norm * head_size + perp * head_size * 0.4
            head2 = end - dir_norm * head_size - perp * head_size * 0.4

            triangle = Polygon([end, head1, head2], closed=True,
                              facecolor=color, edgecolor=color)
            ax.add_patch(triangle)


def plot_complex(
    c: complex,
    ax: Optional[plt.Axes] = None,
    active: bool = True,
    figsize: Tuple[int, int] = (6, 6)
) -> plt.Axes:
    """
    Visualize a complex number as a point on a circle with arrows.

    Shows the complex number's decomposition into parallel and
    perpendicular components.

    Parameters
    ----------
    c : complex
        The complex number to visualize.
    ax : plt.Axes, optional
        Existing axes to plot on. If None, creates new figure.
    active : bool
        Controls arrow direction convention.
    figsize : Tuple[int, int]
        Figure size if creating new figure.

    Returns
    -------
    plt.Axes
        The matplotlib axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    r = abs(c) / 2
    alpha = np.angle(c)

    # Circle
    pts = 100
    angles = np.linspace(0, 2 * np.pi, pts)
    circle = r * np.column_stack([np.cos(angles), np.sin(angles)])
    ax.plot(circle[:, 0], circle[:, 1], 'k:', linewidth=1)

    # Diameter
    diameter = r * np.array([[-1, 1], [0, 0]])
    ax.plot(diameter[0], diameter[1], 'k:', linewidth=1)

    # Rotated diameter
    rot_matrix = np.array([
        [np.cos(2 * alpha), -np.sin(2 * alpha)],
        [np.sin(2 * alpha), np.cos(2 * alpha)]
    ])
    rot_diameter = rot_matrix @ diameter

    # Arrow styles based on angle
    blue_solid = -np.pi/2 < alpha < np.pi/2
    red_solid = not (-np.pi < alpha < 0)

    # Blue arrow (parallel component)
    if active:
        blue_start = np.array([rot_diameter[0, 0], rot_diameter[1, 0]])
        blue_end = np.array([diameter[0, 1], diameter[1, 1]])
    else:
        blue_start = np.array([diameter[0, 0], diameter[1, 0]])
        blue_end = np.array([rot_diameter[0, 1], rot_diameter[1, 1]])

    _draw_labeled_arrow(ax, blue_start, blue_end, COLOR_REAL,
                       '$2u_\\parallel$', solid=blue_solid, is_real=True)

    # Imaginary arrow (perpendicular component)
    if active:
        imag_start = np.array([rot_diameter[0, 0], rot_diameter[1, 0]])
        imag_end = np.array([diameter[0, 0], diameter[1, 0]])
    else:
        imag_start = np.array([diameter[0, 0], diameter[1, 0]])
        imag_end = np.array([rot_diameter[0, 0], rot_diameter[1, 0]])

    _draw_labeled_arrow(ax, imag_start, imag_end, COLOR_IMAG,
                       '$2u_\\perp$', solid=red_solid, is_real=False)

    # u vector
    ax.annotate('', xy=(rot_diameter[0, 0], rot_diameter[1, 0]),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(rot_diameter[0, 0] * 0.6, rot_diameter[1, 0] * 0.6 + r * 0.15,
            '$\\mathbf{u}$', fontsize=12)

    # Formatting
    ax.set_aspect('equal')
    ax.set_xlim([-1.1 * r, 1.1 * r])
    ax.set_ylim([-1.1 * r, 1.1 * r])
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_xticks([-r, r])
    ax.set_yticks([-r, r])
    ax.set_xticklabels(['$-u$', '$u$'])
    ax.set_yticklabels(['$-u$', '$u$'])
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    return ax


def _planar_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: Union[Tuple[float, float, float], str],
    linestyle: str,
    head_style: str
) -> None:
    ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linestyle=linestyle, linewidth=2)
    if head_style == "ellipse":
        radius = np.linalg.norm(end - start) * 0.02
        ax.add_patch(Circle(end, radius, facecolor=color, edgecolor=color))
    else:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="->", color=color, linewidth=2)
        )


def plot_planar_chord_panel(
    angle: float,
    ax: Optional[plt.Axes] = None,
    label_angle: str = r"$\vartheta$",
    red_negative: bool = False,
    label_offsets: Tuple[float, float] = (0.08, 0.12),
    font_sizes: Tuple[int, int, int] = (12, 11, 12)
) -> plt.Axes:
    """Plot a single planar chord panel matching gen_planar_chords.m."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    title_size, label_size, angle_size = font_sizes
    R = 1.0
    corner_size = 0.08
    arc_r = 0.25
    pts = 100
    angles = np.linspace(0, 2 * np.pi, pts)

    A = np.array([-R, 0.0])
    B = np.array([R, 0.0])
    Q = R * np.array([np.cos(angle), np.sin(angle)])

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    blue = color_cycle[0] if color_cycle else "tab:blue"
    red = color_cycle[1] if len(color_cycle) > 1 else "tab:red"

    ax.plot(R * np.cos(angles), R * np.sin(angles), "k", linewidth=1.5)
    ax.plot([A[0], B[0]], [A[1], B[1]], "k-", linewidth=1)

    d1 = R * np.array([np.cos(angle), np.sin(angle)])
    d2 = -d1
    ax.plot([d2[0], d1[0]], [d2[1], d1[1]], "k:", linewidth=1.2)

    _planar_arrow(ax, Q, A, blue, "-", "plain")
    red_style = "--" if red_negative else "-"
    red_head = "ellipse" if red_negative else "plain"
    _planar_arrow(ax, Q, B, red, red_style, red_head)

    ax.plot(0, 0, "ko", markersize=4, markerfacecolor="w")

    ax.text(A[0] - 0.12, A[1] + label_offsets[0], r"$P_-$", fontsize=label_size, ha="right")
    ax.text(B[0] + 0.08, B[1] + label_offsets[0], r"$P_+$", fontsize=label_size)
    ax.text(Q[0] + label_offsets[1], Q[1] + label_offsets[1], r"$Q$", fontsize=label_size)

    mid_blue = (A + Q) / 2
    ax.text(mid_blue[0] + (- 0.4 if not red_negative else +0.05), mid_blue[1] + 0.05,
            r"$2\cos\frac{\vartheta}{2}$", fontsize=label_size - 1, color=blue)
    mid_red = (B + Q) / 2
    red_label = r"$-2\sin\frac{\vartheta}{2}$" if red_negative else r"$2\sin\frac{\vartheta}{2}$"
    ax.text(mid_red[0] + (0.08 if not red_negative else -0.22),
            mid_red[1] + (0.12 if not red_negative else -0.18),
            red_label, fontsize=label_size - 1, color=red)

    v1 = (A - Q) / np.linalg.norm(A - Q)
    v2 = (B - Q) / np.linalg.norm(B - Q)
    corner1 = Q + corner_size * v1
    corner2 = Q + corner_size * v2
    corner_mid = Q + corner_size * (v1 + v2)
    ax.plot([corner1[0], corner_mid[0]], [corner1[1], corner_mid[1]], "k", linewidth=1)
    ax.plot([corner2[0], corner_mid[0]], [corner2[1], corner_mid[1]], "k", linewidth=1)

    arc_angles = np.linspace(0, angle, 50)
    ax.plot(arc_r * np.cos(arc_angles), arc_r * np.sin(arc_angles), "k", linewidth=1)
    ax.text(arc_r * 1.6 * np.cos(angle / 2), arc_r * 1.6 * np.sin(angle / 2),
            label_angle, fontsize=angle_size)

    ax.axis([-1.5 * R, 1.5 * R, -1.5 * R, 1.5 * R])
    ax.axis("off")
    ax.set_aspect("equal")
    if title_size > 0:
        ax.set_title("", fontsize=title_size)

    return ax


def plot_planar_chords_combined(
    theta: float = np.deg2rad(50.0),
    figsize: Tuple[int, int] = (10, 4)
) -> Tuple[plt.Figure, np.ndarray]:
    """Generate the two-panel planar chord construction figure."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_planar_chord_panel(theta, ax=axes[0], label_angle=r"$\vartheta$", red_negative=False)
    plot_planar_chord_panel(theta + np.pi, ax=axes[1], label_angle=r"$\vartheta+\pi$", red_negative=True)
    fig.tight_layout()
    return fig, axes


def _draw_labeled_arrow(
    ax: plt.Axes,
    start: np.ndarray,
    end: np.ndarray,
    color: str,
    label: str,
    solid: bool = True,
    is_real: bool = True
) -> None:
    """Draw an arrow with a label.

    Real part arrows get arrowheads, imaginary part arrows do not.
    Solid=True for positive, solid=False for negative.
    """
    direction = end - start
    if np.linalg.norm(direction) < 1e-10:
        return

    linestyle = '-' if solid else '--'
    # Arrowhead only for real part
    head_style = '->' if is_real else '-'

    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=head_style, color=color,
                               linestyle=linestyle, lw=2))

    mid = (start + end) / 2 + 0.05 * np.array([1, 0])
    ax.text(mid[0], mid[1], label, fontsize=10, color=color)


def create_interactive_3d(
    spinor: Union[Spinor, np.ndarray],
    show_rotation_axis: Optional[np.ndarray] = None,
    width: int = 700,
    height: int = 700,
    show_legend: bool = True
) -> go.Figure:
    """
    Create an interactive 3D Plotly visualization of a spinor.

    Parameters
    ----------
    spinor : Spinor or np.ndarray
        The spinor to visualize.
    show_rotation_axis : np.ndarray, optional
        If provided, shows this rotation axis.
    width, height : int
        Figure dimensions in pixels.
    show_legend : bool
        If True (default), shows the legend on the right side.

    Returns
    -------
    go.Figure
        Interactive Plotly figure.
    """
    if isinstance(spinor, Spinor):
        s = spinor.components
    else:
        s = np.asarray(spinor).flatten()

    # Extract spinor parameters
    half_theta, half_alpha, half_phi = spinor_to_halfangles(s)

    phi = half_phi * 2
    alpha = half_alpha * 2
    sp_length = np.linalg.norm(s)
    sphere_radius = sp_length ** 2
    s_up = s[0] * sp_length * 2
    s_down = s[1] * sp_length * 2

    up_eq_angle = phi
    down_eq_angle = phi + np.pi

    up_and_equator = sphere_radius * np.array([np.cos(up_eq_angle), np.sin(up_eq_angle), 0])
    down_and_equator = sphere_radius * np.array([np.cos(down_eq_angle), np.sin(down_eq_angle), 0])
    up_and_down = up_and_equator + np.abs(s_up) * np.array([
        np.cos(half_theta) * np.cos(down_eq_angle),
        np.cos(half_theta) * np.sin(down_eq_angle),
        np.sin(half_theta)
    ])

    fig = go.Figure()

    # Equator circle
    n_pts = 50
    angles = np.linspace(0, 2 * np.pi, n_pts)
    circle_pts = sphere_radius * np.column_stack([np.cos(angles), np.sin(angles), np.zeros(n_pts)])
    fig.add_trace(go.Scatter3d(
        x=circle_pts[:, 0], y=circle_pts[:, 1], z=circle_pts[:, 2],
        mode='lines', line=dict(color='black', width=3),
        name='Equator'
    ))

    # Upper plane-dihedral face
    if _is_displayable(np.array([s_up])):
        up_circle, up_x, rot_pt_up = _inclined_face_3d(
            up_and_equator, up_and_down, alpha + phi, is_up=True
        )
        fig.add_trace(go.Scatter3d(
            x=up_circle[:, 0], y=up_circle[:, 1], z=up_circle[:, 2],
            mode='lines', line=dict(color=COLOR_UP, width=6),
            name='Upper plane-dihedral face (↑)'
        ))
        # Diameter
        fig.add_trace(go.Scatter3d(
            x=up_x[:, 0], y=up_x[:, 1], z=up_x[:, 2],
            mode='lines', line=dict(color='black', width=2),
            name='Diameter'
        ))

    # Lower plane-dihedral face
    if _is_displayable(np.array([s_down])):
        down_circle, down_x, rot_pt_down = _inclined_face_3d(
            down_and_equator, up_and_down, -(alpha - phi), is_up=False
        )
        fig.add_trace(go.Scatter3d(
            x=down_circle[:, 0], y=down_circle[:, 1], z=down_circle[:, 2],
            mode='lines', line=dict(color=COLOR_DOWN, width=3),
            name='Lower plane-dihedral face (↓)'
        ))

    # Key points
    if _is_displayable(np.array([s_up])) and _is_displayable(np.array([s_down])):
        fig.add_trace(go.Scatter3d(
            x=[up_and_down[0]], y=[up_and_down[1]], z=[up_and_down[2]],
            mode='markers', marker=dict(size=8, color='black'),
            name='Shared point'
        ))
        fig.add_trace(go.Scatter3d(marker_symbol="triangle_up",
            x=[up_and_equator[0]], y=[up_and_equator[1]], z=[up_and_equator[2]],
            mode='markers', marker=dict(size=8, color=COLOR_UP),
            name='Upper equator point'
        ))
        fig.add_trace(go.Scatter3d(marker_symbol="triangle_down",
            x=[down_and_equator[0]], y=[down_and_equator[1]], z=[down_and_equator[2]],
            mode='markers', marker=dict(size=8, color=COLOR_DOWN),
            name='Lower equator point'
        ))

    # Bloch vector
    vec = np.real(spinor_to_bloch_vector(s))
    fig.add_trace(go.Cone(
        x=[0], y=[0], z=[0],
        u=[vec[1]], v=[vec[2]], w=[vec[3]],
        colorscale=[[0, COLOR_BLOCH], [1, COLOR_BLOCH]],
        showscale=False,
        name='Bloch vector'
    ))

    # Rotation axis if provided
    if show_rotation_axis is not None:
        rot_axis = normalize(show_rotation_axis)
        factor = sphere_radius * 1.2
        fig.add_trace(go.Scatter3d(
            x=[0, factor * rot_axis[0]],
            y=[0, factor * rot_axis[1]],
            z=[0, factor * rot_axis[2]],
            mode='lines',
            line=dict(color='green', width=4, dash='dot'),
            name='Rotation axis'
        ))

    # Layout - axes from -1 to 1, no ticks
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                range=[-1, 1],
                showgrid=False,
                showbackground=False,
                zeroline=False,
                showticklabels=False,
                ticks='',
                title='X'
            ),
            yaxis=dict(
                range=[-1, 1],
                showgrid=False,
                showbackground=False,
                zeroline=False,
                showticklabels=False,
                ticks='',
                title='Y'
            ),
            zaxis=dict(
                range=[0, 1],  # Only show Z >= 0
                showgrid=False,
                showbackground=False,
                zeroline=False,
                showticklabels=False,
                ticks='',
                title='Z'
            ),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.5)  # Adjust aspect for half Z range
        ),
        width=width,
        height=height,
        title='Spinor Plane-Dihedral Visualization',
        showlegend=show_legend,
        legend=dict(
            yanchor='middle',
            y=0.5,
            xanchor='left',
            x=1.02,
            bgcolor='rgba(255, 255, 255, 0.9)'
        )
    )

    return fig
