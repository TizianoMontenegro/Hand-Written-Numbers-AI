import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import project
import base64
import cv2

# Sample test image data
TEST_IMAGE = np.zeros((28, 28), dtype=np.uint8)

@pytest.fixture(autouse=True)
def reset_model():
    """Reset model state before each test"""
    project._model = None

def test_load_model():
    """Test model loading functionality"""
    with patch('project.models.load_model') as mock_load:
        project.load_model("test_model.keras")
        mock_load.assert_called_once_with("test_model.keras")
        assert project._model is not None

def test_preprocess_image():
    """Test image preprocessing pipeline"""
    # Create test base64 image (black square)
    _, img_encoded = cv2.imencode('.png', TEST_IMAGE)
    img_base64 = base64.b64encode(img_encoded).decode('utf-8')
    
    # Test with data URL
    data_url = f"data:image/png;base64,{img_base64}"
    processed = project.preprocess_image(data_url)
    
    # Verify output shape and properties
    assert processed.shape == (1, 28, 28, 1)
    assert processed.dtype == np.float32
    assert np.max(processed) <= 1.0
    assert np.min(processed) >= 0.0

def test_predict_digit():
    """Test digit prediction functionality"""
    # Setup mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1]*9 + [0.9]])  # Predict digit 9
    project._model = mock_model
    
    # Create test input
    test_input = np.zeros((1, 28, 28, 1))
    digit = project.predict_digit(test_input)
    
    # Verify predictions
    assert digit == 9
    mock_model.predict.assert_called_once_with(test_input)