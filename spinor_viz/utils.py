"""
Utility functions for spinor visualization.

This module provides helper functions used across the spinor_viz package.
"""

import numpy as np
from typing import Tuple, Optional


def normalize(v: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.

    Parameters
    ----------
    v : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Unit vector in the same direction, or the original
        vector if its norm is too small.
    """
    v = np.asarray(v)
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v
    return v / n


def rotation_matrix_3d(axis: np.ndarray, angle: float) -> np.ndarray:
    """
    Compute the 3x3 rotation matrix for rotation around an axis.

    Uses Rodrigues' rotation formula.

    Parameters
    ----------
    axis : np.ndarray
        Unit vector for the rotation axis.
    angle : float
        Rotation angle in radians.

    Returns
    -------
    np.ndarray
        3x3 rotation matrix.
    """
    axis = normalize(axis)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


def spherical_to_cartesian(r: float, theta: float, phi: float) -> np.ndarray:
    """
    Convert spherical to Cartesian coordinates.

    Parameters
    ----------
    r : float
        Radial distance.
    theta : float
        Polar angle (from z-axis), in radians.
    phi : float
        Azimuthal angle (in xy-plane from x-axis), in radians.

    Returns
    -------
    np.ndarray
        [x, y, z] Cartesian coordinates.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])


def cartesian_to_spherical(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert Cartesian to spherical coordinates.

    Parameters
    ----------
    x, y, z : float
        Cartesian coordinates.

    Returns
    -------
    Tuple[float, float, float]
        (r, theta, phi) - radial distance, polar angle, azimuthal angle.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    if r < 1e-10:
        return 0.0, 0.0, 0.0

    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    return r, theta, phi


def complex_to_rgb(z: complex, max_mag: Optional[float] = None) -> Tuple[float, float, float]:
    """
    Map a complex number to an RGB color.

    Uses hue for phase and saturation/value for magnitude.

    Parameters
    ----------
    z : complex
        Complex number.
    max_mag : float, optional
        Maximum magnitude for normalization. If None, uses |z|.

    Returns
    -------
    Tuple[float, float, float]
        (r, g, b) color values in [0, 1].
    """
    import colorsys

    mag = abs(z)
    phase = np.angle(z)

    # Map phase to hue [0, 1]
    hue = (phase + np.pi) / (2 * np.pi)

    # Map magnitude to saturation and value
    if max_mag is None:
        max_mag = mag if mag > 0 else 1

    sat = 1.0
    val = min(mag / max_mag, 1.0)

    return colorsys.hsv_to_rgb(hue, sat, val)


def linspace_circle(n_points: int = 50) -> np.ndarray:
    """
    Generate evenly spaced angles for drawing a circle.

    Parameters
    ----------
    n_points : int
        Number of points.

    Returns
    -------
    np.ndarray
        Array of angles from 0 to 2π.
    """
    return np.linspace(0, 2 * np.pi, n_points)


def generate_sphere_mesh(
    n_theta: int = 20,
    n_phi: int = 30,
    radius: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate mesh grid for a sphere surface.

    Parameters
    ----------
    n_theta : int
        Number of latitude divisions.
    n_phi : int
        Number of longitude divisions.
    radius : float
        Sphere radius.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (X, Y, Z) mesh grids for the sphere surface.
    """
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)

    THETA, PHI = np.meshgrid(theta, phi)

    X = radius * np.sin(THETA) * np.cos(PHI)
    Y = radius * np.sin(THETA) * np.sin(PHI)
    Z = radius * np.cos(THETA)

    return X, Y, Z
