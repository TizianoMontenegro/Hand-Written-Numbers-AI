from flask import Flask, request, jsonify, render_template
import numpy as np
import base64
# from tensorflow.python.keras import models
from keras import models
import cv2

app = Flask(__name__)
model = models.load_model("model.h5")  # Load saved model

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json().get("image")
    image_data = base64.b64decode(data.split(",")[1])
    
    # Read image with OpenCV
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    # Invert colors (black background, white digit)
    img = cv2.bitwise_not(img)
    
    # Resize and center digit
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Normalize and reshape for model
    img = img.astype("float32") / 255.0
    img = img.reshape(1, 28, 28, 1)

    prediction = model.predict(img)
    predicted_digit = int(np.argmax(prediction))

    return jsonify({"digit": predicted_digit})
