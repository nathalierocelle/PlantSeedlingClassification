import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
import joblib
from preprocessing_predictions import preprocess_data,prediction

model = joblib.load(r'cnn_model_3.pkl')

st.set_page_config(page_icon="🌱")

def main():
    st.write("""
         # Image Classification
         """
         )
    file = st.file_uploader("Upload the image to be classified U0001F447", type=["jpg", "png"])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    if file is None:
    st.text("Please upload an image file")
    else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    
    img = preprocess_data(file)
    image_class, score = prediction(img)
    st.write("The image is classified as",image_class)
    st.write("The similarity score is approximately",score)
    print("The image is classified as ",image_class, "with a similarity score of",score)
           

if __name__ == '__main__':
    main()