import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
from Pre_Processing import skeleton_int
import pandas as pd
import numpy as np
import sys

"""
This module extracts Histogram of Oriented Gradients (HOG) features from character images.
HOG features capture local shape information by calculating gradient directions in local image regions.
"""

def HOG_Training(processed_image, df, i):
    """
    Extracts HOG features from training images.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image
        df (pandas.DataFrame): DataFrame to store the features
        i (int): Index of current image in DataFrame
        
    Returns:
        None. HOG features are stored in DataFrame column 'hog_fv'
    """
    hog_fv, hog_image = hog(processed_image, 
                           orientations=9,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           block_norm='L2', 
                           visualize=True, 
                           channel_axis=None)

    df.at[i,'hog_fv'] = hog_fv


def HOG_Input(processed_image, df, i):
    """
    Extracts HOG features from input images.
    Uses same parameters as training to maintain consistency.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image
        df (pandas.DataFrame): DataFrame to store the features
        i (int): Index of current image in DataFrame
        
    Returns:
        None. HOG features are stored in DataFrame column 'hog_fv'
    """
    hog_fv, hog_image = hog(processed_image, 
                           orientations=9,
                           pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2),
                           block_norm='L2', 
                           visualize=True, 
                           channel_axis=None)

    df.at[i,'hog_fv'] = hog_fv


def plot_HOG(processed_image, hog_image):
    """
    Visualizes the original image and its HOG feature representation side by side.
    
    Args:
        processed_image (numpy.ndarray): Original pre-processed image
        hog_image (numpy.ndarray): HOG visualization image
        
    Returns:
        None. Displays the plot using matplotlib.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
    
    ax1.axis('off')
    ax1.imshow(processed_image, cmap=plt.cm.gray)
    ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.axis('off')
    ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    ax2.set_title('Histogram of Oriented Gradients')
    plt.show()