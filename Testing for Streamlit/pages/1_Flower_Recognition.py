import streamlit as st
from PIL import Image
import numpy as np
import joblib
import pandas as pd

st.set_page_config(page_title="Flower Recognition", page_icon=":bouquet:", layout="wide")

st.markdown("# Flower Recognition")

# Create a file uploader component in Streamlit that accepts multiple image files
uploaded_files = st.file_uploader("Choose your files...", type=["jpg","jpeg"], accept_multiple_files=True)

# Load the pre-trained model
model = joblib.load('ResNet50V2_model_final.joblib')

# Define the classes
class_names = ['Babi', 'Calimerio', 'Chrysanthemum', 'Hydrangeas', 'Lisianthus', 'Pingpong', 'Rosy', 'Tana']

# Define the predict function
# Define the predict function
def predict(file_obj):
    img = Image.open(file_obj)
    img = img.resize((256, 256))  # resize to (256, 256)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.0
    predictions = model.predict(img_array)
    score = np.squeeze(predictions)
    return score

# Display uploaded images
if uploaded_files is not None:
    st.write("Uploaded Images:")
    row = st.empty()
    num_cols = 4
    num_rows = int(np.ceil(len(uploaded_files) / num_cols))
    uploaded_images = []
    for i in range(num_rows):
        cols = row.columns(num_cols)
        for j in range(num_cols):
            index = i * num_cols + j
            if index < len(uploaded_files):
                file_obj = uploaded_files[index]
                image = Image.open(file_obj)
                filename = file_obj.name
                image = image.resize((256, 256))
                cols[j].image(image, caption=filename, use_column_width=True)
                uploaded_images.append((file_obj, filename))

    # Submit button for classification
    if st.button("Submit"):
        st.write("Results:")
        data = []
        for uploaded_image in uploaded_images:
            file_obj, filename = uploaded_image
            score = predict(file_obj)
            prediction = f"These flowers are likely {class_names[np.argmax(score)]} with a {100 * np.max(score)}% confidence."
            data.append([filename, prediction])

        # Show new results in table
        df = pd.DataFrame(data, columns=['File Name', 'Result'])
        st.table(df)