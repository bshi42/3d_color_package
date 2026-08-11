"""Regression tests for the NumPy part of atlas Procrustes alignment."""

import numpy as np

from .shape_bridge import _row_kabsch_rotation


def _proper_rotation(seed=0):
    matrix = np.random.default_rng(seed).normal(size=(3, 3))
    rotation, _ = np.linalg.qr(matrix)
    if np.linalg.det(rotation) < 0:
        rotation[:, -1] *= -1
    return rotation


def test_row_kabsch_recovers_known_rotation():
    source = np.random.default_rng(10).normal(size=(32, 3))
    source -= source.mean(axis=0)
    expected = _proper_rotation(seed=11)

    actual = _row_kabsch_rotation(source, source @ expected)

    # For row-vector points, U @ Vt recovers R.  The transposed convention
    # used by the old implementation fails this check for a generic rotation.
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(source @ actual, source @ expected, atol=1e-12)
    np.testing.assert_allclose(actual.T @ actual, np.eye(3), atol=1e-12)
    assert np.linalg.det(actual) > 0.0


def test_row_kabsch_rejects_reflection():
    source = np.random.default_rng(12).normal(size=(32, 3))
    source -= source.mean(axis=0)
    reflected = source @ np.diag([1.0, 1.0, -1.0])

    actual = _row_kabsch_rotation(source, reflected)

    assert np.linalg.det(actual) > 0.0
    # A proper rotation cannot reproduce a reflection exactly, but it should
    # still be no worse than leaving the source unaligned.
    aligned_error = np.linalg.norm(source @ actual - reflected)
    identity_error = np.linalg.norm(source - reflected)
    assert aligned_error < identity_error
