import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# =============================
#   CONFIG
# =============================
DATASET_PATH = "augmented_dataset"
IMAGE_SIZE = (224, 224)     # MobileNetV2 default size
BATCH_SIZE = 16
EPOCHS = 20
LR = 0.0001                 # low learning rate for fine-tuning

# =============================
#   DATASET LOADER
# =============================

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.20
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

# =============================
#   MOBILE NET V2 BASE MODEL
# =============================

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False   # Freeze weights

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)

preds = Dense(train_data.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
#   TRAINING
# =============================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)

# =============================
#   SAVE MODEL
# =============================
model.save("model_skin_mobilenet2.h5")
print("\nModel berhasil disimpan sebagai model_skin_mobilenet.h5!")

# =============================
#   GRAPHING
# =============================

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Accuracy")
plt.legend(["Training", "Validation"])

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Loss")
plt.legend(["Training", "Validation"])

plt.show()
