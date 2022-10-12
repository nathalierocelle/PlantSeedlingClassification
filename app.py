import streamlit as st
import cv2
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from tensorflow import keras
from preprocessing_predictions import preprocess_data,prediction


model = joblib.load(r'cnn_model_3.pkl')
labels = ['Black-grass','Charlock','Cleavers','Common Chickweed','Common wheat',
          'Fat Hen','Loose Silky-bent','Maize','Scentless Mayweed','Shepherd’s Purse',
          'Small-flowered Cranesbill','Sugar beet']

df = pd.DataFrame(['Black-grass',
 'Charlock',
 'Cleavers',
 'Common Chickweed',
 'Common wheat',
 'Fat Hen',
 'Loose Silky-bent',
 'Maize',
 'Scentless Mayweed',
 'Shepherd’s Purse',
 'Small-flowered Cranesbill',
 'Sugar beet'],columns=['Species'])

st.set_page_config(page_icon="🌱")
st.image('Banner.jpg')

def main():
    st.sidebar.markdown("<h2>About the app</h2>", unsafe_allow_html=True)
    st.sidebar.write("""
            The app aims to determine the species of a seedling from an image.
             """)
    st.sidebar.write("""
            🌿Disclaimer: As of today, the following are the types of plant seedlings that were
            used in training the model: 
             """)
    st.sidebar.write(df) 
    
    file = st.file_uploader(type=["jpg", "png"])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    if file is None:
        st.text("🍂Please upload your plant image")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        st.write("Your plant is classified as",file.getvalue())
        img = preprocess_data(file.getvalue())
        image_class, score = prediction(model,img,labels)
    
        st.write("Your plant is classified as",image_class)
        st.write("The similarity score is approximately",score)
    #print("The image is classified as ",image_class, "with a similarity score of",score)
           

if __name__ == '__main__':
    main()