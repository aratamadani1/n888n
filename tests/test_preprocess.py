import pytest

np = pytest.importorskip("numpy")

from facesense.agent import FaceSenseAgent


def test_preprocess_face_grayscale_constant():
    face = np.full((70, 50), 255, dtype=np.uint8)
    tensor = FaceSenseAgent.preprocess_face(face)
    assert tensor.shape == (1, 1, 64, 64)
    expected_value = (1.0 - FaceSenseAgent._MODEL_MEAN) / FaceSenseAgent._MODEL_STD
    assert np.allclose(tensor, expected_value, atol=1e-4)


def test_preprocess_face_color_conversion():
    rgb = np.zeros((64, 64, 3), dtype=np.uint8)
    rgb[..., 2] = 255  # Pure blue should become a dark grayscale value.
    tensor = FaceSenseAgent.preprocess_face(rgb)
    expected_gray = 0.114 * 1.0  # Normalised blue channel contribution.
    expected_value = (expected_gray - FaceSenseAgent._MODEL_MEAN) / FaceSenseAgent._MODEL_STD
    # Check a couple of representative pixels.
    assert np.allclose(tensor[0, 0, 0, 0], expected_value, atol=1e-4)
    assert np.allclose(tensor[0, 0, -1, -1], expected_value, atol=1e-4)


def test_resize_to_square_preserves_corners():
    image = np.arange(16, dtype=np.float32).reshape(4, 4)
    resized = FaceSenseAgent._resize_to_square(image, 2)
    assert resized.shape == (2, 2)
    assert np.isclose(resized[0, 0], image[0, 0])
    assert np.isclose(resized[-1, -1], image[-1, -1])


def test_softmax_normalisation():
    logits = np.array([0.0, 1.0, 2.0])
    probs = FaceSenseAgent.softmax(logits)
    assert probs.shape == (3,)
    assert np.all(probs > 0)
    assert np.isclose(np.sum(probs), 1.0)
