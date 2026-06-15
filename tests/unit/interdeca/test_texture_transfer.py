"""Texture transfer tests."""

import imageio.v2 as imageio
import numpy as np

from color_deca.interdeca.core.texture_transfer import TextureTransferService, calculate_average_texture


def test_calculate_average_texture_preserves_uint8_truncation(tmp_path):
    imageio.imwrite(tmp_path / "a.png", np.array([[[0, 1, 2]]], dtype=np.uint8))
    imageio.imwrite(tmp_path / "b.png", np.array([[[1, 2, 5]]], dtype=np.uint8))

    output = calculate_average_texture(tmp_path)

    assert output == tmp_path / "average_texture.png"
    observed = imageio.imread(output)
    np.testing.assert_array_equal(observed, np.array([[[0, 1, 3]]], dtype=np.uint8))


def test_calculate_average_texture_returns_existing_average_without_recomputing(tmp_path):
    imageio.imwrite(tmp_path / "a.png", np.array([[[0, 0, 0]]], dtype=np.uint8))
    imageio.imwrite(tmp_path / "average_texture.png", np.array([[[9, 9, 9]]], dtype=np.uint8))

    output = TextureTransferService().calculate_average_texture(tmp_path)

    observed = imageio.imread(output)
    np.testing.assert_array_equal(observed, np.array([[[9, 9, 9]]], dtype=np.uint8))
