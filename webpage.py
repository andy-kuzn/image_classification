import io
import time
import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications import EfficientNetB7
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import \
    preprocess_input, decode_predictions


def load_image():
    """Creating form for uploading image"""
    uploaded_file = st.file_uploader(label="""Choose an image file
                                     for uploading""", key=9876543)
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data, caption="""I had to shrink this image
                 a little so I can process it faster!""", width=600)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


@st.cache(allow_output_mutation=True)
def load_model():
    """Loading image classification model from Keras"""
    model = EfficientNetB7(weights='imagenet')
    return model


def preprocess_image(img):
    """Image preprocessing function"""
    img = img.resize((600, 600))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def print_predictions(preds):
    """Printing top 5 classes and their probabilities"""
    classes = decode_predictions(preds, top=5)[0]
    for cl in classes:
        st.write(cl[1], cl[2])
        

# Web page layout
model = load_model()
st.title('Image Classification')
st.subheader("*Find out what's on your image with Keras EfficientNetB7*")
img = load_image()
# Show Classify button only if image has been loaded.
if img != None:
    result = st.button('**Classify this image** :face_with_monocle:')
    if result:
        x = preprocess_image(img)
        preds = model.predict(x)
        with st.spinner('This may take a second or two...'):
            time.sleep(3)
        st.write("""**Top 5 classification results and their
                 probabilities:**""")
        print_predictions(preds)
