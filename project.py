import numpy as np
import cv2
import base64
from tf_keras import models

_model = None  # Global model instance

def load_model(model_path="model.keras"):
    """Load the trained Keras model from disk"""
    global _model
    if _model is None:
        _model = models.load_model(model_path)
    return _model

def preprocess_image(image_base64):
    """Preprocess base64 image data for model prediction"""
    if ',' in image_base64:
        image_base64 = image_base64.split(',')[1]
    image_data = base64.b64decode(image_base64)
    # Read image with OpenCV
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    # Invert colors (black background, white digit)
    img = cv2.bitwise_not(img)
    # Resize and center digit
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # Normalize and reshape for model
    img = img.astype("float32") / 255.0
    return img.reshape(1, 28, 28, 1)

def predict_digit(image_array):
    """Predict digit from preprocessed image array"""
    if _model is None:
        load_model()
    predictions = _model.predict(image_array)
    return int(np.argmax(predictions))

def main():
    """Main entry point for the application"""
    load_model()
    from app import app
    app.run(debug=True)

if __name__ == "__main__":
    main()
