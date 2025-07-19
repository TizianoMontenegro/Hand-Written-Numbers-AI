from flask import Flask, request, jsonify, render_template
from project import preprocess_image, predict_digit

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json().get("image")
    try:
        img_array = preprocess_image(data)
        digit = predict_digit(img_array)
        return jsonify({"digit": digit})
    except Exception as e:
        return jsonify({"error": str(e)}), 400