import pathlib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras import layers, Model

base = pathlib.Path(r"C:\Users\Bhime\Desktop\dataset\cats_dogs_project")
train_dir = base / "dataset" / "train"

IMG_SIZE = 160
BATCH = 32

def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image, label

# Auto split from train folder - no val folder needed
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH,
    validation_split=0.2, subset="training", seed=123
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=(IMG_SIZE,IMG_SIZE), batch_size=BATCH,
    validation_split=0.2, subset="validation", seed=123
)

print("Classes:", train_ds.class_names) # Should be ['cats', 'dogs']

augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomBrightness(0.3),
    layers.RandomContrast(0.2),
])

train_ds = train_ds.map(lambda x,y: (augment(x, training=True), y))
train_ds = train_ds.map(preprocess).cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).cache().prefetch(tf.data.AUTOTUNE)

base_model = MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE,IMG_SIZE,3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = Model(inputs, outputs)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("--- Phase 1: Training top layer ---")
model.fit(train_ds, validation_data=val_ds, epochs=10)

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='binary_crossentropy', metrics=['accuracy'])
print("--- Phase 2: Fine-tuning ---")
model.fit(train_ds, validation_data=val_ds, epochs=10)

model.save(base / "cats_dogs_model_V2_FIXED.h5")
print("SAVED: cats_dogs_model_V2_FIXED.h5 - NOW TEST WHITE DOG!")