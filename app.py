import streamlit as st
import cv2
from PIL import Image
import numpy as np
import joblib
from tensorflow import keras
from preprocessing_predictions import preprocess_data,prediction

model = joblib.load(r'cnn_model_3.pkl')
labels = ['Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat',
          'Fat Hen','Loose Silky-bent','Maize','Scentless Mayweed','Shepherd’s Purse',
          'Small-flowered Cranesbill','Sugar beet']

st.set_page_config(page_icon="🌱")
st.image('Banner.jpg')

def main():
    st.sidebar.markdown("<h2>About the app</h2>", unsafe_allow_html=True)
    st.sidebar.write("""
            The app aims to classify the seedling type of your plant
             """)
    file = st.file_uploader("Upload your plant image to be classified", type=["jpg", "png"])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    if file is None:
        st.text("Please upload an image file")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
    
        img = preprocess_data(file)
        image_class, score = prediction(model,img,labels)
    
        st.write("The image is classified as",image_class)
        st.write("The similarity score is approximately",score)
    #print("The image is classified as ",image_class, "with a similarity score of",score)
           

if __name__ == '__main__':
    main()