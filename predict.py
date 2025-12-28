import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Sesuaikan urutan label dengan training terakhir
class_names = ["combination", "dry", "normal", "oily"]

# Load model terbaik kamu
model = tf.keras.models.load_model("model_skin_mobilenet2.h5")

def predict_skin(img_path):
    # MobileNetV2 input size
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    
    # Normalisasi khusus MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(img_array)[0]
    class_id = np.argmax(prediction)
    confidence = float(prediction[class_id])

    print("Jenis kulit :", class_names[class_id])
    print("Keyakinan   :", round(confidence * 100, 2), "%")

# Contoh penggunaan
predict_skin("contoh.jpg")
