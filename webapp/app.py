import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from eda import plot_color_distribution
from eda import plot_size_distribution
from eda import plot_category_distribution
from eda import flower_dataset_directory

st.sidebar.title("Explanatory Data Analysis")

chart_selection = st.sidebar.radio('', ['Color channel distribution', 'Image size distribution', 'Flower Category Distribution'])

# Render selected chart
if chart_selection == 'Color channel distribution':
    st.title('Color channel distribution (percentage) of flower images')
    plot_color_distribution(flower_dataset_directory,10,3)
elif chart_selection == 'Image size distribution':
    st.title('Image Size Distribution')
    plot_size_distribution(flower_dataset_directory,10,6)
elif chart_selection == 'Flower Category Distribution':
    st.title('Flower Category Distribution')
    plot_category_distribution(flower_dataset_directory)