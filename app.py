import os
import keras 
import numpy as np
from keras.models import load_model
import streamlit as st
import tensorflow as tf

st.header('Flower Classification CNN Model ')
flower_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
model = load_model('Flower_Reco_model.keras')

def classify_images(image_path):
    input_image = tf.keras.utils.load_img(image_path,target_size=(180,180))
    input_image_array = tf.keras.utils.img_to_array(input_image)
    input_image_exp_dim = tf.expand_dims(input_image_array,axis=0)

    prediction = model.predict(input_image_exp_dim)
    result = tf.nn.softmax(prediction[0])
    outcome = 'The Image belongs to ' + flower_names[np.argmax(result)] + ' with a score of '+ str(np.max(result)*100)
    return outcome

uploaded_file =st.file_uploader('Upload an image')
if uploaded_file is not None:
    with open(os.path.join('upload',uploaded_file.name),'wb') as f:
        f.write(uploaded_file.getbuffer())

        st.image(uploaded_file,width=200)
st.markdown(classify_images(uploaded_file))       