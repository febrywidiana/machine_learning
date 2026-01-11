from flask import Flask, render_template, request, send_from_directory
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

app = Flask(__name__)

model = tf.keras.models.load_model("model_skin_fixed.keras")


class_names = ["Dry", "Normal", "Oily"]

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def predict_skin(img_path):
    # Setup gambar ke 224x224
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Preprocessing wajib MobileNetV2
    img_array = preprocess_input(img_array)

    # Prediksi
    prediction = model.predict(img_array)[0]
    class_id = np.argmax(prediction)
    confidence = prediction[class_id]

    return class_names[class_id], confidence

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", result=None)
            
        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", result=None)

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            label, prob = predict_skin(filepath)

            result = {
                "label": label,
                "confidence": round(float(prob) * 100, 2),
                "image": file.filename
            }

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)