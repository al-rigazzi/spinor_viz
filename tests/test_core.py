"""
Tests for spinor_viz.core module.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from spinor_viz.core import (
    pauli,
    spinor_rotation,
    spinor_to_halfangles,
    halfangles_to_spinor,
    spinor_to_bloch_vector,
    Spinor,
    normalize,
)


class TestPauliMatrices:
    """Tests for Pauli matrices."""

    def test_pauli_0_is_identity(self):
        """σ₀ should be the identity matrix."""
        assert_allclose(pauli(0), np.eye(2))

    def test_pauli_matrices_hermitian(self):
        """All Pauli matrices should be Hermitian."""
        for i in range(4):
            sigma = pauli(i)
            assert_allclose(sigma, sigma.conj().T)

    def test_pauli_matrices_square_to_identity(self):
        """σᵢ² = I for i = 1, 2, 3."""
        for i in range(1, 4):
            sigma = pauli(i)
            assert_allclose(sigma @ sigma, np.eye(2))

    def test_pauli_commutation(self):
        """Test [σ₁, σ₂] = 2iσ₃ and cyclic permutations."""
        s1, s2, s3 = pauli(1), pauli(2), pauli(3)

        assert_allclose(s1 @ s2 - s2 @ s1, 2j * s3)
        assert_allclose(s2 @ s3 - s3 @ s2, 2j * s1)
        assert_allclose(s3 @ s1 - s1 @ s3, 2j * s2)

    def test_pauli_invalid_index(self):
        """Should raise error for invalid index."""
        with pytest.raises(ValueError):
            pauli(4)
        with pytest.raises(ValueError):
            pauli(-1)


class TestSpinorRotation:
    """Tests for spinor rotation operations."""

    def test_identity_rotation(self):
        """Zero rotation should give identity."""
        R = spinor_rotation(np.array([0, 0, 0]))
        assert_allclose(R, np.eye(2), atol=1e-10)

    def test_rotation_is_unitary(self):
        """Rotation matrices should be unitary (R†R = I)."""
        for _ in range(10):
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis) * np.random.rand() * 2 * np.pi
            R = spinor_rotation(axis)
            assert_allclose(R.conj().T @ R, np.eye(2), atol=1e-10)

    def test_rotation_is_special_unitary(self):
        """Rotation matrices should have determinant 1."""
        for _ in range(10):
            axis = np.random.randn(3)
            R = spinor_rotation(axis)
            assert_allclose(np.linalg.det(R), 1.0, atol=1e-10)

    def test_z_rotation_on_spin_up(self):
        """Z-rotation on |↑⟩ should only add phase."""
        angle = np.pi / 3
        R = spinor_rotation(np.array([0, 0, angle]))
        result = R @ np.array([1, 0])

        expected = np.array([np.exp(-1j * angle / 2), 0])
        assert_allclose(result, expected, atol=1e-10)

    def test_x_rotation_by_pi(self):
        """180° rotation around x should swap |↑⟩ and |↓⟩ (up to phase)."""
        R = spinor_rotation(np.array([np.pi, 0, 0]))

        up_rotated = R @ np.array([1, 0])
        down_rotated = R @ np.array([0, 1])

        # |↑⟩ → -i|↓⟩
        assert_allclose(np.abs(up_rotated[0]), 0, atol=1e-10)
        assert_allclose(np.abs(up_rotated[1]), 1, atol=1e-10)

        # |↓⟩ → -i|↑⟩
        assert_allclose(np.abs(down_rotated[0]), 1, atol=1e-10)
        assert_allclose(np.abs(down_rotated[1]), 0, atol=1e-10)


class TestHalfAngles:
    """Tests for half-angle conversions."""

    def test_spin_up_halfangles(self):
        """Spin up should have θ/2 = 0."""
        s = np.array([1, 0])
        half_theta, _, _ = spinor_to_halfangles(s)
        assert_allclose(half_theta, 0, atol=1e-10)

    def test_spin_down_halfangles(self):
        """Spin down should have θ/2 = π/2."""
        s = np.array([0, 1])
        half_theta, _, _ = spinor_to_halfangles(s)
        assert_allclose(half_theta, np.pi / 2, atol=1e-10)

    def test_roundtrip_conversion(self):
        """Converting to halfangles and back should give original spinor."""
        for _ in range(10):
            # Random spinor
            s_original = np.random.randn(2) + 1j * np.random.randn(2)
            s_original = s_original / np.linalg.norm(s_original)

            # Convert to halfangles and back
            ht, ha, hp = spinor_to_halfangles(s_original)
            s_reconstructed = halfangles_to_spinor(ht, ha, hp)

            # They should be equal up to a global phase
            ratio = s_original / s_reconstructed
            if np.abs(s_reconstructed[0]) > 1e-10:
                phase = ratio[0]
            else:
                phase = ratio[1]

            assert_allclose(np.abs(phase), 1, atol=1e-10)


class TestBlochVector:
    """Tests for Bloch vector computation."""

    def test_spin_up_bloch(self):
        """Spin up should point to +z."""
        s = np.array([1, 0])
        v = spinor_to_bloch_vector(s)
        assert_allclose(v[1:], [0, 0, 1], atol=1e-10)

    def test_spin_down_bloch(self):
        """Spin down should point to -z."""
        s = np.array([0, 1])
        v = spinor_to_bloch_vector(s)
        assert_allclose(v[1:], [0, 0, -1], atol=1e-10)

    def test_spin_plus_x_bloch(self):
        """(|↑⟩ + |↓⟩)/√2 should point to +x."""
        s = np.array([1, 1]) / np.sqrt(2)
        v = spinor_to_bloch_vector(s)
        assert_allclose(v[1:], [1, 0, 0], atol=1e-10)

    def test_bloch_vector_normalized(self):
        """Bloch vector should have length 1 for normalized spinors."""
        for _ in range(10):
            s = np.random.randn(2) + 1j * np.random.randn(2)
            s = s / np.linalg.norm(s)
            v = spinor_to_bloch_vector(s)
            assert_allclose(np.linalg.norm(v[1:]), 1, atol=1e-10)


class TestSpinorClass:
    """Tests for the Spinor class."""

    def test_spin_up_creation(self):
        """Test creation of spin up state."""
        s = Spinor.spin_up()
        assert_allclose(s.components, [1, 0])

    def test_spin_down_creation(self):
        """Test creation of spin down state."""
        s = Spinor.spin_down()
        assert_allclose(s.components, [0, 1])

    def test_from_angles(self):
        """Test creation from spherical angles."""
        s = Spinor.from_angles(0, 0)
        assert_allclose(s.bloch_vector(), [0, 0, 1], atol=1e-10)

        s = Spinor.from_angles(np.pi, 0)
        assert_allclose(s.bloch_vector(), [0, 0, -1], atol=1e-10)

    def test_rotation_preserves_norm(self):
        """Rotation should preserve spinor norm."""
        s = Spinor.random()
        original_norm = s.norm

        for _ in range(10):
            axis = np.random.randn(3)
            s_rot = s.rotate(axis)
            assert_allclose(s_rot.norm, original_norm, atol=1e-10)

    def test_4pi_periodicity(self):
        """Spinor should return to original after 4π rotation."""
        s = Spinor.random()
        axis = normalize(np.random.randn(3))

        # After 2π: s → -s
        s_2pi = s.rotate_axis_angle(axis, 2 * np.pi)
        ratio_2pi = s_2pi.components / s.components
        if np.abs(s.up) > 1e-10:
            assert_allclose(ratio_2pi[0], -1, atol=1e-8)

        # After 4π: s → s
        s_4pi = s.rotate_axis_angle(axis, 4 * np.pi)
        ratio_4pi = s_4pi.components / s.components
        if np.abs(s.up) > 1e-10:
            assert_allclose(ratio_4pi[0], 1, atol=1e-8)

    def test_inner_product(self):
        """Test inner product ⟨s|t⟩."""
        s = Spinor.spin_up()
        t = Spinor.spin_down()

        # Orthogonal states
        assert_allclose(s @ t, 0, atol=1e-10)

        # Self inner product
        assert_allclose(s @ s, 1, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
