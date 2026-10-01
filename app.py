import pathlib
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

base = pathlib.Path(r"C:\Users\Bhime\Desktop\dataset\cats_dogs_project")
MODEL_PATH = base / "cats_dogs_model_V2_FIXED.h5"

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

st.set_page_config(page_title="Cat vs Dog AI", page_icon="🐶")
st.title("🐱 Cat vs 🐶 Dog Classifier V2")
st.write("Accuracy: 98.3% - Fixed for white dogs")

file = st.file_uploader("Upload image", type=["jpg","jpeg","png"])

if file:
    st.image(file, caption="Uploaded Image", width='stretch')
    img = load_img(file, target_size=(160,160))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)

    pred = model.predict(arr)[0][0]

    if pred < 0.5:
        st.success(f"**It's a CAT** 🐱 - {(1-pred)*100:.1f}% confident")
    else:
        st.success(f"**It's a DOG** 🐶 - {pred*100:.1f}% confident")
    st.caption(f"Raw score: {pred:.4f} (0=Cat, 1=Dog)")