import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image, ImageOps
import numpy as np

@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('93.h5')
    return model

model = load_model()

st.write("# Braille Classification System")

file = st.file_uploader("Choose braille photo from computer", type=["jpg", "png"])

#def import_and_predict(image_data, model):
    #size = (28, 28,3)
    #image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    #image = ImageOps.grayscale(image)
    #img = np.asarray(image)
    #img = img.reshape((size[0], size[1], 1))  
   #img_reshape = img[np.newaxis, ...]
    #prediction = model.predict(img_reshape)
    #return prediction
def import_and_predict(image_data, model):
    size = (28, 28)
    
    image -ImageOPs.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_NEAREST)
    
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    img_reshape = img.reshape((1,) + img.shape + (1,))
    
    predictiction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    class_names=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 
                 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
                 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 
                 'Y', 'Z']
    string = "OUTPUT: " + class_names[np.argmax(prediction)]
    st.success(string)
