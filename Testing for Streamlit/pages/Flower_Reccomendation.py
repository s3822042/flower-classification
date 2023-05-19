import streamlit as st
from PIL import Image
import numpy as np
import os
import base64
from io import BytesIO
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

def recall_cnn(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    possible_positives = np.sum(np.round(np.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + np.finfo(float).eps)
    return recall

def precision_cnn(y_true, y_pred):
    true_positives = np.sum(np.round(np.clip(y_true * y_pred, 0, 1)))
    predicted_positives = np.sum(np.round(np.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + np.finfo(float).eps)
    return precision

def f1_cnn(y_true, y_pred):
    precision = precision_cnn(y_true, y_pred)
    recall = recall_cnn(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + np.finfo(float).eps))

METRICS = ["accuracy", recall_cnn, precision_cnn, f1_cnn]

def recommend_similar_images(image, num_images=12):
    # Define the input shape expected by the model
    input_shape = (224, 224)

    # Resize the uploaded image to match the input shape expected by the model
    image_resized = image.resize(input_shape)

    # Convert the resized image to an array
    img_array_resized = np.array(image_resized)

    # Normalize the image
    img_array_resized = img_array_resized.astype('float32') / 255.0

    # Load the pre-trained model using joblib
    VGG16_model = joblib.load('VGG16_model.joblib')

    # Create a new model that outputs the desired layer's activations
    intermediate_layer_model = Model(inputs=VGG16_model.input, outputs=VGG16_model.get_layer('flatten').output)

    # Extract the feature vector of the uploaded image
    features = intermediate_layer_model.predict(np.expand_dims(img_array_resized, axis=0))

    # Calculate cosine similarities between the uploaded image and all the images in the dataset
    similarities = []
    folder_path = os.path.join('data', 'All type flowers')
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg'):
            image_path_i = os.path.join(folder_path, file_name)
            img_i = Image.open(image_path_i).resize(input_shape)
            x_i = np.array(img_i)
            x_i = np.expand_dims(x_i, axis=0)
            x_i = tf.keras.applications.vgg16.preprocess_input(x_i)
            features_i = intermediate_layer_model.predict(x_i)
            similarity = cosine_similarity(features, features_i)[0][0]
            similarities.append((image_path_i, similarity))

    # Sort the similarities in descending order and select the top_n images
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[:num_images]

    # Extract the similar images for display
    similar_images = []
    for image_path, similarity in similarities:
        similar_images.append((image_path, similarity))

    return similar_images

# Set page configuration
st.set_page_config(page_title="Flower Recommendation", page_icon=":bouquet:")

st.markdown("# Flower Recommendation")

# Create a file uploader component in Streamlit
uploaded_file = st.file_uploader('Upload an image', type=['jpg', 'jpeg'])

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Resize the image if its width or height is greater than 500 pixels
    max_size = 500
    if image.width > max_size or image.height > max_size:
        image.thumbnail((max_size, max_size))

    # Convert the uploaded image to a data URL
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    uploaded_image_data_url = base64.b64encode(buffered.getvalue()).decode()

    # Create an HTML table to display:
    table_html = f'''
    <style>
        table {{
            background-color: white;
        text-align: center;
        vertical-align: middle;
        }}
    </style>
    <table>
        <tr>
            <th>File Name</th>
                <td>{uploaded_file.name}</td>
        </tr>
        <tr>
            <th>Image</th>
                <td><img src="data:image/jpeg;base64,{uploaded_image_data_url}" /></td>
        </tr>
    </table>
    '''

    # Display the table in Streamlit
    st.markdown(table_html, unsafe_allow_html=True)

    # Get similar images based on the uploaded image using the pre-trained VGG16 model
    similar_images = recommend_similar_images(image)

    st.write('The flower you are looking for:')
    
    cols_per_row = 4
    image_size = (350, 350)  # Fixed size for recommendation images

    for i in range(0, len(similar_images), cols_per_row):
        cols = st.columns(cols_per_row)
        for j in range(cols_per_row):
            if i + j < len(similar_images):
                similar_image_path, similarity = similar_images[i + j]
                with open(similar_image_path, 'rb') as f:
                    similar_image_data_url = base64.b64encode(f.read()).decode()
                    flower_name = os.path.basename(similar_image_path).split('_')[0].capitalize()
                    title_text = f'Flower: {flower_name}\nSimilarity: {similarity:.2f}'
                    cols[j].image(
                        Image.open(similar_image_path).resize(image_size),
                        use_column_width=True
                    )
                    cols[j].markdown(f'<div style="text-align:center"><a href="{similar_image_data_url}" title="{title_text}">{flower_name}</a></div>', unsafe_allow_html=True)
#CSS Style background for pages
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://github.com/s3822042/flower-classification/blob/Zoe/Testing%20for%20Streamlit/bg.jpg?raw=true");
background-size: 180%;
background-position: top left;
background-repeat: no-repeat;
background-attachment: local;
}}

/*tittle page color*/
span.css-10trblm.e16nr0p30 {{
    color: #A61635;
}}

/*file name uploaded text color*/
div.uploadedFileName.css-1uixxvy{{
    color: white;
}}

div.css-10ix4kq {{
    color: white;
}}

small.css-1aehpvj{{
    color: #ff7a7a;
}}

/*text color*/
p{{
  color: white;
  font-weight: bold;
}}

div.block-container.css-1y4p8pa.egzxvld4{{
    padding-right: 100px;
    padding-top: 50px;
    width: 100%;
}}

div.css-4u6e0b {{
    background-color: rgba(0, 0, 0, 0.5);
}}

ul.css-lrlib {{
    background-color: #272525;
}}

section.css-vjj2ce {{
    background-color: #BE3838;
    color: rgb(49, 51, 63);
}}

span.css-9ycgxx {{
    color:white;
}}

[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}

[data-testid=stSidebar] {{
    background-color: #B13746;
}}

div.css-j7qwjs {{
    background-color: #BDBBBB;
}}

/*image*/
img {{
    cursor: pointer;
    transition: all .2s ease-in-out;
}}

img:hover {{
    transform: scale(1.1);
}}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)