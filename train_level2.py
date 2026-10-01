import pathlib
import tensorflow as tf

base = pathlib.Path(r"C:\Users\Bhime\Desktop\dataset\cats_dogs_project")
train_dir = base/"dataset"/"train"
test_dir = base/"dataset"/"test"

# 1. Strong Augmentation
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir, image_size=(160,160), batch_size=32
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir, image_size=(160,160), batch_size=32
)

augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

# Apply
train_ds = train_ds.map(lambda x,y: (augment(x), y))

# 2. Build Model same as before
base_model = tf.keras.applications.MobileNetV2(input_shape=(160,160,3), include_top=False, weights='imagenet')
base_model.trainable = False

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_ds, validation_data=test_ds, epochs=5)
model.save(base/"cats_dogs_model_level2.h5")
print("Level 2 Model Saved")



print("Starting Fine-Tuning Task 2...")

base_model.trainable = True
for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

model.fit(train_ds, validation_data=test_ds, epochs=5)
model.save(base/"cats_dogs_model_FINAL_97.h5")