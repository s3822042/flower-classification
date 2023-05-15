import streamlit as st
from PIL import Image
import numpy as np
import joblib
import base64
from io import BytesIO

st.set_page_config(page_title="Flower Recognition", page_icon=":bouquet:")

st.markdown("# Flower Recognition")

# Create a file uploader component in Streamlit that accepts only one image file
uploaded_file = st.file_uploader("Choose your file...", type=["jpg", "jpeg"])

# Load the pre-trained model
model = joblib.load('ResNet50V2_model_final.joblib')

# Define the classes
class_names = ['Babi', 'Calimerio', 'Chrysanthemum', 'Hydrangeas', 'Lisianthus', 'Pingpong', 'Rosy', 'Tana']

# Define the predict function
def predict(file_obj):
    img = Image.open(file_obj)
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    predictions = model.predict(img_array)
    score = np.squeeze(predictions)
    return score

# Check if a file has been uploaded
if uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Make a prediction using your pre-trained model and predict function
    prediction = predict(uploaded_file)
    predicted_class = class_names[np.argmax(prediction)]
    predicted_class_index = np.argmax(prediction)
    predicted_probability = prediction[predicted_class_index]

    # Convert the image to a data URL
    buffered = BytesIO()
    image.save(buffered, format='JPEG')
    image_data_url = base64.b64encode(buffered.getvalue()).decode()

    # Create an HTML table to display the image, its file name, and the classification result
    table_html = f'''
        <table>
            <tr>
                <th>File Name</th>
                <td>{uploaded_file.name}</td>
            </tr>
            <tr>
                <th>Image</th>
                <td><img src="data:image/jpeg;base64,{image_data_url}" /></td>
            </tr>
            <tr>
                <th>Classification</th>
                <td>{predicted_class} ({predicted_probability:.2f})</td>
            </tr>
        </table>
    '''

    # Display the table in Streamlit
    st.markdown(table_html, unsafe_allow_html=True)