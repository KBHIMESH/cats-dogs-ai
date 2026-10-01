import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

MODEL_PATH = "cats_dogs_model_V2_FIXED.h5"

@st.cache_resource
def load_my_model():
    return load_model(MODEL_PATH)

model = load_my_model()

# PAGE CONFIG
st.set_page_config(
    page_title="Cat vs Dog AI - Bhimesh",
    page_icon="🐶",
    layout="centered"
)

# HEADER
st.markdown("""
<style>
.big-font {font-size:30px!important; font-weight:700;}
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([3,1])
with col1:
    st.title("🐱 Cat vs 🐶 Dog Classifier V2")
    st.caption("MobileNetV2 | 98.3% Accuracy | Fixed for white dogs")
with col2:
    st.link_button("⭐ GitHub", "https://github.com/KBHIMESH/cats-dogs-ai", use_container_width=True)

st.divider()

file = st.file_uploader("📤 Upload a cat or dog image", type=["jpg","jpeg","png"])

if file:
    c1, c2 = st.columns(2)
    with c1:
        st.image(file, caption="Uploaded Image", use_container_width=True)

    # Predict
    img = load_img(file, target_size=(160,160))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    pred = float(model.predict(arr)[0][0])

    cat_conf = (1-pred)*100
    dog_conf = pred*100

    with c2:
        st.subheader("Result")
        if pred < 0.5:
            st.success(f"### It's a CAT 🐱")
            st.metric("Confidence", f"{cat_conf:.1f}%")
            st.progress(int(cat_conf))
        else:
            st.success(f"### It's a DOG 🐶")
            st.metric("Confidence", f"{dog_conf:.1f}%")
            st.progress(int(dog_conf))

        st.caption(f"Raw score: {pred:.4f} (0=Cat, 1=Dog)")
        st.bar_chart({"CAT 🐱": cat_conf, "DOG 🐶": dog_conf})

    if max(cat_conf, dog_conf) > 90:
        st.balloons()

else:
    st.info("👆 Try it with a white dog - the old model used to fail, this V2 FIXED version works!")

st.divider()
st.caption("Built by Bhimesh | TensorFlow + Streamlit | Model: cats_dogs_model_V2_FIXED.h5 (23 MB)")