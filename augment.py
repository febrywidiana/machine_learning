import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil

SOURCE_DIR = "dataset"
OUTPUT_DIR = "augmented_dataset"

# Buat ulang folder output biar bersih
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# Augment yang lebih kuat & natural
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.10,
    height_shift_range=0.10,
    zoom_range=0.20,
    shear_range=0.10,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    fill_mode="nearest"
)

# Jumlah augment per gambar
AUG_PER_IMAGE = 12   # naikkan supaya dataset besar

IMG_SIZE = (224, 224)  # MobileNetV2 input

for class_name in os.listdir(SOURCE_DIR):
    class_dir = os.path.join(SOURCE_DIR, class_name)
    if not os.path.isdir(class_dir):
        continue

    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    print(f"[AUG] Processing class: {class_name}")

    for img_name in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_name)

        # Simpan gambar asli juga → dataset lebih kuat
        shutil.copy(img_path, os.path.join(output_class_dir, img_name))

        # Load image
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape)

        save_prefix = os.path.splitext(img_name)[0]

        # Generate augment
        i = 0
        for batch in datagen.flow(
            img_array,
            batch_size=1,
            save_to_dir=output_class_dir,
            save_prefix=save_prefix,
            save_format="jpg"
        ):
            i += 1
            if i >= AUG_PER_IMAGE:
                break

print("\n=== Augment selesai! Dataset baru ada di folder 'augmented_dataset' ===")
