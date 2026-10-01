import pathlib
import tensorflow as tf

base = pathlib.Path(r"C:\Users\Bhime\Desktop\dataset\cats_dogs_project")
train_dir = base/"dataset"/"train"
test_dir = base/"dataset"/"test"

train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=(160,160), batch_size=32
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=(160,160), batch_size=32
)

# Task 3: CNN from Scratch - NO MobileNetV2
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(160,160,3)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary() # See params - only ~500k params vs MobileNetV2 3.5 Million

model.fit(train_ds, validation_data=test_ds, epochs=5)
model.save(base/"cats_dogs_scratch_75.h5")
print("Scratch Model Saved - 75% only")