import streamlit as st
import base64
from pathlib import Path
from fastai.vision.all import *

def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

img = get_img_as_base64("pict.jpg")

# Add custom CSS to set the background image
st.markdown(
    f"""
    <style>
    [data-testid="stAppViewContainer"]{{
    background-image: url('data:image/png;base64,{img}');
    background-size: cover;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit app
st.title("Pneumonia Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file:
    st.image(uploaded_file)
    #image convertion
    img = Image.open(uploaded_file)
    # do stuff with `img`

    output = io.BytesIO()
    img.save(output, format='JPEG')  # or another format
    output.seek(0)

    #model
    model = load_learner('pneumonia_classifier.pkl')

    #prediction

    pred, pred_id, probs =  model.predict(img)
    st.success(f'Prediction: {pred}')
    st.info(f'Accuracy: {probs[pred_id]*100:.1f}%')