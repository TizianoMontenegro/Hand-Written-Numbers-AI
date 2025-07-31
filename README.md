# DRecognizer

Watch the video
[![Watch the video](https://github.com/user-attachments/assets/e400f8e1-1edc-4b3a-b0ea-ff2123a9823d)](https://youtu.be/2urnHt8TROY)


## Project Structure

* **app.py**: Defines a Flask web app that serves the user interface and a `/predict` endpoint. The home route renders `index.html`, while the predict route receives the drawn image data and returns the predicted digit.
* **train.py**: A Python script using TensorFlow/Keras to build and train a Convolutional Neural Network (CNN) on the MNIST digit dataset. After training (with data augmentation), it saves the trained model to `model.keras` for later use.
* **project.py**: Contains helper functions for loading the saved model, preprocessing images, and performing predictions. The `preprocess_image` function decodes a base64 PNG, inverts colors, resizes it to 28×28 pixels, and normalizes pixel values. The `predict_digit` function runs the preprocessed image through the model to find the most likely digit.
* **test\_project.py**: A test suite (using `pytest`) that checks these functions. It mocks the model to verify `load_model`, `preprocess_image`, and `predict_digit` behave correctly with known inputs.
* **templates/index.html**: The HTML page that provides the drawing interface. It includes instructions and a canvas for the user to draw a digit. When the user clicks **Predict**, it explains how the image is converted into a 28×28 grid, scanned by the AI, and the resulting digit is displayed.
* **static/**: Contains front-end assets:

  * `style.css` and images/fonts for page styling.
  * `canvasDraw.js`: JavaScript code that handles drawing on the `<canvas>`, clearing it, and sending the drawn image to the `/predict` endpoint via a POST request.
* **requirements.txt**: Lists dependencies (`flask`, `tensorflow`, `numpy`, `opencv-python-headless`) needed to run the app.
* **model.keras**: The saved Keras model file produced by `train.py`, which is loaded at runtime to make predictions.

## CNN Model Training

1. **Data Loading**: The script loads the MNIST dataset of handwritten digits (60,000 training images and 10,000 test images).
2. **Network Architecture**: It defines a CNN with multiple convolutional and max-pooling layers to extract visual features, followed by dense layers for classification. For example, the network includes convolutional blocks (Conv2D → BatchNorm → Conv2D → MaxPooling → Dropout) with 32 and 64 filters, then a Conv2D with 128 filters, and finally a fully connected layer of size 256 before the 10-way softmax output.
3. **Data Augmentation**: To improve generalization, it uses `ImageDataGenerator` with random rotations, zooms, and shifts of the training images.
4. **Training**: The model is compiled with the Adam optimizer and trained (up to 50 epochs) on the augmented data. Callbacks like `EarlyStopping` and `ReduceLROnPlateau` prevent overfitting and adjust the learning rate. Finally, the trained model is saved in TensorFlow’s native format (`model.keras`).

## Image Preprocessing & Prediction

* **Loading the Model**: When the app starts or a prediction is requested, `project.load_model()` loads `model.keras` into a global `_model` variable (only once) so it can be used for inference.
* **Preprocessing**: The drawn image arrives as a base64-encoded PNG. The function `preprocess_image` decodes this data, reads it into a grayscale image using OpenCV (`cv2.imdecode`), and **inverts colors** (making the digit white on a black background). It then resizes the image to 28×28 pixels (the MNIST input size) and normalizes pixel values to the \[0,1] range.
* **Prediction**: The preprocessed 28×28 array is fed into the CNN model. The function `predict_digit` calls the model’s `predict` method and takes the index of the highest output probability as the predicted digit. This integer digit is returned to the web app as JSON.

## Web Application Workflow

1. **User Interface**: The user navigates to the home page (`index.html`). A canvas element is presented where the user can draw a digit (0–9). The page includes instructions and a **Predict** button.
2. **Drawing on Canvas**: The `canvasDraw.js` script initializes the canvas with a white background and listens for mouse events to draw black strokes as the user draws. The user can also click **Clear** to reset the canvas.
3. **Sending for Prediction**: When the user clicks **Predict**, the canvas image is converted to a base64 PNG (`toDataURL`) and sent via a POST request to `/predict` as JSON.
4. **Server Prediction**: On the server side, Flask receives the image data in the `/predict` route. It uses `preprocess_image` to convert it to a model-ready array and `predict_digit` to find the digit.
5. **Result Display**: The predicted digit is sent back to the browser, and JavaScript updates the page to show the result (the “Prediction” field is updated with the digit). The page even includes a user-friendly description: *“the AI — trained on thousands of handwritten digits — scans \[the] 28×28 grid … calculates which number matches best, and instantly returns the result”*.

## Testing and Validation

* **Automated Tests**: The `test_project.py` uses `pytest` to ensure each component works as expected. It resets the model state, mocks the model loading, and checks that `preprocess_image` outputs a 28×28 float array, and that `predict_digit` correctly identifies the highest probability class. This gives confidence that the core logic is correct.
* **Manual Testing**: Since this is a user-facing app, a final check involves drawing digits in the browser to verify real-time predictions.
