import streamlit as st
import base64
import pathlib
from fastai.vision.all import *
import platform
plt = platform.system()
if plt == 'Linux':
    pathlib.Windows = pathlib.PosixPath

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

# Check if an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    if st.button("Predict"):
        # Load the trained model
        learn_inf = load_learner('pneumonia_classifier.pkl')

        # Perform prediction
        pred_class, pred_idx, probabilities = learn_inf.predict(image)

        # Format the Result and Probability in the same line
        result_text = f"<font color='white' style='font-size:30px'>Result: </font>"
        if pred_class == "PNEUMONIA":
            result_text += f"<font color='red' style='font-size:30px'>{pred_class}</font>"
        else:
            result_text += f"<font color='green' style='font-size:30px'>{pred_class}</font>"

        # Display Result and Probability in the same line with increased font size
        st.markdown(result_text, unsafe_allow_html=True)
        st.markdown(f"<font style='font-size:30px'>Accuracy: {probabilities[pred_idx]*100:.2f}%</font>", unsafe_allow_html=True)
