import streamlit as st
from fastai.vision.all import *

# Load the trained model
path = Path('')
learn_inf = load_learner(path/'pneumonia_classifier.pkl')

# Streamlit app
st.title("Pneumonia Image Classifier")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
# Display the uploaded image
image = Image.open(uploaded_file)
st.image(image, caption="Uploaded Image.", use_column_width=True)

if st.button("Predict"):
    if uploaded_file is not None:

        # Perform prediction
        pred_class, pred_idx, probabilities = learn_inf.predict(image)

        # Format the Result and Probability in the same line
        result_text = f"<font color='white' style='font-size:30px'>Result: </font>"
        if pred_class == "PNEUMONIA":
            result_text += f"<font color='red' style='font-size:30px'>{pred_class}</font>"
        else:
            result_text += f"<font color='green' style='font-size:3opx'>{pred_class}</font>"

        # Display Result and Probability in the same line with increased font size
        st.markdown(result_text, unsafe_allow_html=True)
        st.markdown(f"<font style='font-size:30px'>Probability: {probabilities[pred_idx]*100:.2f}%</font>", unsafe_allow_html=True)





