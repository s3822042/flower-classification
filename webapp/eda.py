import os
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import cv2

flower_dataset_directory = "data/Flowers/"
flower_category_foldername = os.listdir(flower_dataset_directory)

link = []
for label in flower_category_foldername:
    path = os.path.join(flower_dataset_directory, label) # combine path and labels
    link.append(path) # append in link

def plot_color_distribution(main_folder, figure_width, figure_height):
    sub_folders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

    num_rows = int(np.ceil(len(sub_folders) / 2))
    num_cols = min(len(sub_folders), 4)

    figsize = (figure_width, figure_height * num_rows)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=figsize)
    axs = axs.ravel()

    for i in range(len(sub_folders)):
        sub_folder_name = os.path.basename(sub_folders[i])

        file_names = os.listdir(sub_folders[i])
        file_names = [f for f in file_names if f.endswith(".jpg") or f.endswith(".png")]

        channel_data = np.zeros((256, 3))

        for j in range(len(file_names)):
            img_path = os.path.join(sub_folders[i], file_names[j])
            img = plt.imread(img_path)

            for k in range(3):
                channel_data[:, k] += np.histogram(img[:, :, k], bins=256, range=(0, 256))[0]

        channel_data /= np.sum(channel_data, axis=0)
        channel_data *= 100

        colors = ['red', 'green', 'blue']
        labels = ['Red', 'Green', 'Blue']

        for k in range(3):
            axs[i].plot(range(256), channel_data[:, k], label=labels[k], color=colors[k], linewidth=1)

        axs[i].set_title(sub_folder_name)
        axs[i].legend()

    for i in range(len(sub_folders), len(axs)):
        fig.delaxes(axs[i])

    for ax in axs.flat:
        ax.set_xlabel("Pixel value")
        ax.set_ylabel("Percentage")

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    st.pyplot(fig)

def plot_size_distribution(img_dir, width, height):
    categories = os.listdir(img_dir)
    fig = plt.figure(figsize=(width, height))
    grid_shape = (2, 4)
    for i, category in enumerate(categories):
        size_list = []
        for filename in os.listdir(os.path.join(img_dir, category)):
            img_path = os.path.join(img_dir, category, filename)
            img = cv2.imread(img_path)
            size = os.path.getsize(img_path)
            size_list.append(size)

        ax = fig.add_subplot(grid_shape[0], grid_shape[1], i+1)
        ax.hist(size_list, bins=50)
        ax.set_title(category)
        ax.set_xlabel('Image Size (Bytes)')
        ax.set_ylabel('Frequency')
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    st.pyplot(fig)

def plot_category_distribution(main_dir):
    categories = os.listdir(main_dir)
    category_counts = [len(os.listdir(os.path.join(main_dir, cat))) for cat in categories]

    fig, ax = plt.subplots()
    ax.bar(categories, category_counts)
    ax.set_xticklabels(categories, rotation=45)
    ax.set_xlabel('Categories')
    ax.set_ylabel('Number of Images')

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    st.pyplot(fig)