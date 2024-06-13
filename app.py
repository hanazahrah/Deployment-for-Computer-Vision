import streamlit as st
from PIL import Image
from classification import classification_img
import cv2
import numpy as np

st.title("Image Detection")
st.header("Bukti Tf")
st.text("Upload image")

uploaded_file = st.file_uploader("Choose image...", type=['jpg','png','jpeg'])
print(uploaded_file)
if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    #image = Image.open(uploaded_file)
    #st.image(image, caption='uploaded image', use_column_width=True)
    st.image(opencv_image, channels="BGR")
    st.write("")
    st.write("Classifying..")
    label = classification_img(opencv_image, 'best_5.pt')
    for i, det in enumerate(label):
        if len(det[:, -1].unique()) == 0:
            st.write('tf asli')
        else:
            st.write('tf palsu')
    #st.write(label)
