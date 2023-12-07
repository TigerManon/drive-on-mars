# Imports
import os

import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from data import load_image_set, decompose_label, create_df
from params import MASK_PATH_TRAIN

custom_colors = {
    0: 'blue',
    1: 'yellow',
    2: 'purple',
    3: 'red',
    4: 'white',
}

cmap_labels = ListedColormap([custom_colors[i] for i in range(max(custom_colors)+1)])

def show_image(image, mask=None):
    """

    """
    n_sub = 1
    if mask is not None:
        n_sub=2
    fig, ax = plt.subplots(1, n_sub, figsize=(16,9))

    ax[0].axis('off')
    ax[0].imshow(image, cmap='gray')
    ax[0].set_title('Raw image')

    if n_sub==2:
        ax[1].axis('off')
        ax[1].imshow(image*mask, cmap='gray')
        ax[1].set_title('Masked')



def show_labels_comp(image,label):
    """

    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={'width_ratios': [2, 2.2]})

    ax[0].axis('off')
    ax[0].imshow(image, cmap='gray', aspect='equal')  # Set aspect='equal' for square
    ax[0].set_title('Original')

    ax[1].axis('off')
    ax[1].imshow(image, cmap='gray', aspect='equal')  # Set aspect='equal' for square
    img = ax[1].imshow(label, alpha=0.3, cmap=cmap_labels, aspect='equal')  # Set aspect='equal' for square
    ax[1].set_title('Labels')

    divider = make_axes_locatable(ax[1])
    cbar_ax = divider.append_axes("right", size="5%", pad=0.2)  # Adjust the size and pad

    # Add colorbar to the right subplot
    cbar = fig.colorbar(img, cax=cbar_ax, ticks=np.unique(label), orientation='vertical')

    # Add tick labels to the colorbar
    tick_labels = labels_key.values()
    cbar.set_ticks(np.unique(label))
    cbar.set_ticklabels(tick_labels)
    cbar.ax.tick_params(axis='y', length=0)
    cbar.ax.set_yticklabels(tick_labels, va='center')


    plt.show()


def show_labels_indv(label,image):
    """

    """

    label_0, label_1, label_2, label_3, label_4 = decompose_label(label)

    fig, ax = plt.subplots(1, 3, figsize=(16,9))
    ax[0].axis('off')
    ax[0].imshow(label_0, cmap='YlOrRd')
    ax[0].set_title('Soil')
    ax[1].axis('off')
    ax[1].imshow(label_1, cmap='YlOrRd')
    ax[1].set_title('Bedrock')
    ax[2].axis('off')
    ax[2].imshow(label_2, cmap='YlOrRd')
    ax[2].set_title('Sand')

    fig, ax = plt.subplots(1, 3, figsize=(16,9))
    ax[0].axis('off')
    ax[0].imshow(label_3, cmap='YlOrRd')
    ax[0].set_title('Big Rocks')
    ax[1].axis('off')
    ax[1].imshow(label_4,cmap='YlOrRd')
    ax[1].set_title('Terrain (Null)')

    ax[2].axis('off')
    ax[2].imshow(image,cmap='gray')
    ax[2].imshow(label, alpha =0.6, cmap='YlOrRd')

    ax[2].set_title('All')


if __name__ == '__main__':

    df_train = create_df(MASK_PATH_TRAIN)

    # Example
    image_id_ex = 456
    image, label, mask = load_image_set(image_id_ex, df_train)

    show_image(image, mask)
    show_labels_indv(label,image)
    show_labels_comp(image,label)
