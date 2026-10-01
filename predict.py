from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("cats_dogs_model.h5")

# CHANGE THIS PATH to your downloaded image
image_path = "camel.jpg" # put your google image name here

img = Image.open(image_path).convert('RGB').resize((160,160))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0) # make it [1,160,160,3]

pred = model.predict(img_array)[0][0]

print(f"Score: {pred}")

if pred > 0.5:
    print(f"Prediction: DOG 🐶 ({pred*100:.2f}% confident)")
else:
    print(f"Prediction: CAT 🐱 ({(1-pred)*100:.2f}% confident)")

# If you test with tiger/lion/elephant - it will still say cat or dog
# because model only knows 2 classes. It will guess closest one.