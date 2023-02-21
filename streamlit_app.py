import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import pandas as pd

@st.cache_data
def load_data():
    class_names = pd.read_csv("class_names.csv")
    class_names = class_names["0"]
    return class_names

def predict_image(image):
    base_model = tf.keras.applications.EfficientNetB0(include_top=False)
    inputs = tf.keras.layers.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(101)(x)
    outputs = tf.keras.layers.Activation("softmax", dtype=tf.float32)(x)
    model = tf.keras.Model(inputs, outputs)
    model.load_weights('my_model_weights.h5')
    image = tf.image.resize(image, [224, 224]).numpy().reshape((1,) + (224, 224, 3))
    pred = model.predict(image)
    certainty = np.max(pred)
    pred = np.argmax(pred)
    pred = class_names[pred]
    return pred, certainty

class_names = load_data()
st.header("DeepFood")
st.write('DeepFood helps you with finding out what you are eating. Upload an image of your food and find with delicacy you have.')
with st.form('image_form'):
    # Add a file uploader to the form
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])
    
    # Add a submit button to the form
    submit_button = st.form_submit_button(label='Submit')
    
    # Make a prediction when the user submits the form
    if submit_button and uploaded_file is not None:
        image = Image.open(uploaded_file)
        prediction, certainty = predict_image(image)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write(f'Prediction: {prediction.title()}. Certainty: {round(certainty * 100, 2)} %')
