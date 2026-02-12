"""
Core spinor mathematics.

This module provides the fundamental mathematical operations for spinors,
including Pauli matrices, rotation operators, and conversions between
different spinor representations.
"""

import numpy as np
from scipy.linalg import expm
from typing import Tuple, Optional, Union
from dataclasses import dataclass


def pauli(n: int) -> np.ndarray:
    """
    Return the n-th Pauli matrix.

    Parameters
    ----------
    n : int
        Index of the Pauli matrix (0, 1, 2, or 3).
        - 0: Identity matrix
        - 1: σ_x (sigma_x)
        - 2: σ_y (sigma_y)
        - 3: σ_z (sigma_z)

    Returns
    -------
    np.ndarray
        2x2 complex Pauli matrix.

    Examples
    --------
    >>> pauli(0)
    array([[1.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j]])
    >>> pauli(3)
    array([[ 1.+0.j,  0.+0.j],
           [ 0.+0.j, -1.+0.j]])
    """
    if n == 0:
        return np.array([[1, 0], [0, 1]], dtype=complex)
    elif n == 1:
        return np.array([[0, 1], [1, 0]], dtype=complex)
    elif n == 2:
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif n == 3:
        return np.array([[1, 0], [0, -1]], dtype=complex)
    else:
        raise ValueError("n must be 0, 1, 2, or 3")


def spinor_rotation(axis: np.ndarray) -> np.ndarray:
    """
    Compute the SU(2) rotation matrix for a given rotation axis.

    The rotation angle is encoded in the magnitude of the axis vector.
    R = exp(-i * (axis · σ) / 2)

    Parameters
    ----------
    axis : np.ndarray
        3D vector representing the rotation axis. The magnitude gives
        the rotation angle in radians.

    Returns
    -------
    np.ndarray
        2x2 complex unitary rotation matrix in SU(2).

    Examples
    --------
    >>> # Rotation by π around z-axis
    >>> R = spinor_rotation(np.array([0, 0, np.pi]))
    >>> R @ np.array([1, 0])  # |↑⟩ → -i|↑⟩
    """
    axis = np.asarray(axis).flatten()
    if len(axis) != 3:
        raise ValueError("axis must be a 3D vector")

    # Construct the generator: axis · σ
    generator = axis[0] * pauli(1) + axis[1] * pauli(2) + axis[2] * pauli(3)

    # R = exp(-i * generator / 2)
    return expm(-1j * generator / 2)


def spinor_to_halfangles(s: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert a spinor to its half-angle representation.

    A spinor s = [s_up, s_down]^T can be parameterized as:
        s_up   = |s| * cos(θ/2) * exp(i(α/2 + φ/2))
        s_down = |s| * sin(θ/2) * exp(i(α/2 - φ/2))

    Parameters
    ----------
    s : np.ndarray
        2-element complex spinor [s_up, s_down].

    Returns
    -------
    Tuple[float, float, float]
        (half_theta, half_alpha, half_phi) - the three half-angles.
        - half_theta: θ/2 ∈ [0, π/2]
        - half_alpha: α/2 (overall phase)
        - half_phi: φ/2 (relative phase)

    Examples
    --------
    >>> s = np.array([1, 0])  # Spin up state
    >>> half_theta, half_alpha, half_phi = spinor_to_halfangles(s)
    >>> half_theta  # Should be 0
    0.0
    """
    s = np.asarray(s).flatten()
    if len(s) != 2:
        raise ValueError("Spinor must be a 2-element vector")

    # half_theta from ratio of magnitudes
    half_theta = np.arctan2(np.abs(s[1]), np.abs(s[0]))

    # Extract phases
    angle_up = np.angle(s[0])
    angle_down = np.angle(s[1])

    # half_alpha and half_phi from phase combinations
    half_alpha = -((angle_up + angle_down) / 2) % (2 * np.pi)
    half_phi = -((angle_up - angle_down) / 2) % (2 * np.pi)

    return half_theta, half_alpha, half_phi


def halfangles_to_spinor(
    half_theta: float,
    half_alpha: float,
    half_phi: float,
    norm: float = 1.0
) -> np.ndarray:
    """
    Convert half-angles to a spinor.

    Inverse of spinor_to_halfangles.

    Parameters
    ----------
    half_theta : float
        θ/2 - determines ratio of up/down components.
    half_alpha : float
        α/2 - overall phase.
    half_phi : float
        φ/2 - relative phase (azimuthal).
    norm : float, optional
        Norm of the output spinor (default 1.0).

    Returns
    -------
    np.ndarray
        2-element complex spinor.
    """
    s_up = norm * np.cos(half_theta) * np.exp(1j * (-half_alpha - half_phi))
    s_down = norm * np.sin(half_theta) * np.exp(1j * (-half_alpha + half_phi))
    return np.array([s_up, s_down], dtype=complex)


def spinor_to_bloch_vector(s: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the Bloch vector representation of a spinor.

    For a spinor |s⟩, computes ⟨t|σ_μ|s⟩ for μ = 0, 1, 2, 3.
    If t is not provided, uses t = s (expectation values).

    Parameters
    ----------
    s : np.ndarray
        2-element spinor.
    t : np.ndarray, optional
        Second spinor for cross-terms. Defaults to s.

    Returns
    -------
    np.ndarray
        4-element vector [⟨σ_0⟩, ⟨σ_1⟩, ⟨σ_2⟩, ⟨σ_3⟩].
        The last 3 components form the Bloch vector.
    """
    s = np.asarray(s).flatten()
    if len(s) != 2:
        raise ValueError("Spinor must be a 2-element vector")

    if t is None:
        t = s
    else:
        t = np.asarray(t).flatten()
        if len(t) != 2:
            raise ValueError("Second spinor must be a 2-element vector")

    v = np.array([
        np.vdot(t, pauli(0) @ s),
        np.vdot(t, pauli(1) @ s),
        np.vdot(t, pauli(2) @ s),
        np.vdot(t, pauli(3) @ s),
    ])

    return v


def fix_gauge(s: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    Fix the gauge of a spinor so that its first component is real and non-negative.

    Divides the spinor by the phase factor of its first component.
    When the first component is (near-)zero, the spinor is returned unchanged.

    This corresponds to the common quantum-computing convention where
    state vectors are written with a real non-negative first amplitude.

    Parameters
    ----------
    s : np.ndarray
        2-element complex spinor.
    tol : float
        Threshold below which the first component is considered zero.

    Returns
    -------
    np.ndarray
        Gauge-fixed spinor with ``s[0]`` real and non-negative (when it
        was not near-zero).

    Examples
    --------
    >>> fix_gauge(np.array([1j, 1]))
    array([1.+0.j, 0.-1.j])
    """
    s = np.asarray(s, dtype=complex).flatten()
    if np.abs(s[0]) < tol:
        return s
    phase = np.exp(1j * np.angle(s[0]))
    return s / phase


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    v = np.asarray(v)
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v
    return v / n


def rotate_vector_around_axis(
    vectors: np.ndarray,
    axis: np.ndarray,
    angle_degrees: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Rotate 3D vectors around an arbitrary axis.

    Uses Rodrigues' rotation formula.

    Parameters
    ----------
    vectors : np.ndarray
        Array of shape (N, 3) or (3,) containing vectors to rotate.
    axis : np.ndarray
        Unit vector defining the rotation axis.
    angle_degrees : float or np.ndarray
        Rotation angle(s) in degrees.

    Returns
    -------
    np.ndarray
        Rotated vectors with same shape as input.
    """
    vectors = np.atleast_2d(vectors)
    axis = normalize(np.asarray(axis))
    angle = np.atleast_1d(np.radians(angle_degrees))

    if len(angle) == 1:
        angle = np.full(len(vectors), angle[0])

    # Rodrigues' rotation formula
    cos_theta = np.cos(angle)[:, np.newaxis]
    sin_theta = np.sin(angle)[:, np.newaxis]

    # v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
    cross = np.cross(axis, vectors)
    dot = np.sum(axis * vectors, axis=1, keepdims=True)

    rotated = vectors * cos_theta + cross * sin_theta + axis * dot * (1 - cos_theta)

    if rotated.shape[0] == 1:
        return rotated[0]
    return rotated


@dataclass
class Spinor:
    """
    A spinor with convenient methods for manipulation and visualization.

    Attributes
    ----------
    components : np.ndarray
        2-element complex array [s_up, s_down].

    Examples
    --------
    >>> s = Spinor.spin_up()
    >>> s.rotate([0, 0, np.pi])  # Rotate by π around z
    >>> s.bloch_vector()
    """

    components: np.ndarray

    def __post_init__(self):
        self.components = np.asarray(self.components, dtype=complex).flatten()
        if len(self.components) != 2:
            raise ValueError("Spinor must have exactly 2 components")

    @classmethod
    def spin_up(cls) -> "Spinor":
        """Create a spin-up state |↑⟩ = [1, 0]^T."""
        return cls(np.array([1, 0], dtype=complex))

    @classmethod
    def spin_down(cls) -> "Spinor":
        """Create a spin-down state |↓⟩ = [0, 1]^T."""
        return cls(np.array([0, 1], dtype=complex))

    @classmethod
    def from_angles(
        cls,
        theta: float,
        phi: float,
        alpha: float = 0.0,
        norm: float = 1.0
    ) -> "Spinor":
        """
        Create a spinor from spherical angles.

        Parameters
        ----------
        theta : float
            Polar angle θ ∈ [0, π].
        phi : float
            Azimuthal angle φ ∈ [0, 2π].
        alpha : float, optional
            Overall phase α (default 0).
        norm : float, optional
            Norm of the spinor (default 1).
        """
        return cls(halfangles_to_spinor(theta/2, alpha/2, phi/2, norm))

    @classmethod
    def from_bloch(cls, x: float, y: float, z: float) -> "Spinor":
        """
        Create a spinor from Bloch sphere coordinates.

        Parameters
        ----------
        x, y, z : float
            Cartesian coordinates on/in the Bloch sphere.
            For a pure state, x² + y² + z² = 1.
        """
        r = np.sqrt(x**2 + y**2 + z**2)
        if r < 1e-10:
            return cls.spin_up()

        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        return cls.from_angles(theta, phi, norm=np.sqrt(r))

    @classmethod
    def random(cls, seed: Optional[int] = None) -> "Spinor":
        """Create a random normalized spinor."""
        rng = np.random.default_rng(seed)
        phases = rng.uniform(0, 2*np.pi, 2)
        theta = rng.uniform(0, np.pi)
        s = np.array([
            np.cos(theta/2) * np.exp(1j * phases[0]),
            np.sin(theta/2) * np.exp(1j * phases[1])
        ])
        return cls(s)

    @property
    def up(self) -> complex:
        """The spin-up component."""
        return self.components[0]

    @property
    def down(self) -> complex:
        """The spin-down component."""
        return self.components[1]

    @property
    def norm(self) -> float:
        """The norm of the spinor."""
        return np.linalg.norm(self.components)

    def normalized(self) -> "Spinor":
        """Return a normalized copy of the spinor."""
        return Spinor(self.components / self.norm)

    def gauge_fixed(self) -> "Spinor":
        """
        Return a copy with the first component made real and non-negative.

        This removes the global phase freedom by dividing by the phase
        of the first component.  When the first component is near-zero
        the spinor is returned unchanged.

        This is the standard convention used in quantum computing where
        state vectors are written with a real, positive first amplitude.
        """
        return Spinor(fix_gauge(self.components))

    def to_halfangles(self) -> Tuple[float, float, float]:
        """Convert to half-angle representation (θ/2, α/2, φ/2)."""
        return spinor_to_halfangles(self.components)

    def bloch_vector(self) -> np.ndarray:
        """
        Compute the Bloch vector (expectation values of Pauli matrices).

        Returns
        -------
        np.ndarray
            3-element real vector on/in the Bloch sphere.
        """
        v = spinor_to_bloch_vector(self.components)
        return np.real(v[1:])  # [⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩]

    def rotate(self, axis: np.ndarray) -> "Spinor":
        """
        Apply a rotation to the spinor.

        Parameters
        ----------
        axis : np.ndarray
            Rotation axis vector. Magnitude is the rotation angle in radians.

        Returns
        -------
        Spinor
            New rotated spinor.
        """
        R = spinor_rotation(axis)
        return Spinor(R @ self.components)

    def rotate_axis_angle(self, axis: np.ndarray, angle: float) -> "Spinor":
        """
        Rotate by a given angle around a given axis.

        Parameters
        ----------
        axis : np.ndarray
            Unit vector for rotation axis.
        angle : float
            Rotation angle in radians.

        Returns
        -------
        Spinor
            New rotated spinor.
        """
        axis = normalize(np.asarray(axis))
        return self.rotate(axis * angle)

    def __repr__(self) -> str:
        return f"Spinor([{self.up:.4f}, {self.down:.4f}])"

    def __array__(self) -> np.ndarray:
        return self.components

    def __matmul__(self, other: "Spinor") -> complex:
        """Inner product ⟨self|other⟩."""
        return np.vdot(self.components, other.components)
