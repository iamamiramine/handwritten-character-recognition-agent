from skimage.measure import profile_line
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""
This module extracts intensity profile features from character images.
Profiles are taken along horizontal and vertical lines through the center of the image
to capture character shape information.
"""

def Profile_Training(processed_image, df, i):
    """
    Extracts profile features from training images.
    Takes intensity profiles along the horizontal and vertical center lines.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image (64x64)
        df (pandas.DataFrame): DataFrame to store the features
        i (int): Index of current image in DataFrame
        
    Returns:
        None. Features are stored in DataFrame columns 'profile_top_fv' and 'profile_right_fv'
    """
    # Top (horizontal) profile
    start_top = (0, 32)  # (x,y) coordinates
    end_top = (64, 32)
    profile_top = profile_line(processed_image, start_top, end_top, linewidth=5)

    # Right (vertical) profile
    start_right = (32, 0)  # (x,y) coordinates
    end_right = (32, 64)
    profile_right = profile_line(processed_image, start_right, end_right, linewidth=5)

    df.at[i,'profile_top_fv'] = profile_top
    df.at[i,'profile_right_fv'] = profile_right


def Profile_Input(processed_image, df, i):
    """
    Extracts profile features from input images.
    Uses same parameters as training to maintain consistency.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image (64x64)
        df (pandas.DataFrame): DataFrame to store the features
        i (int): Index of current image in DataFrame
        
    Returns:
        None. Features are stored in DataFrame columns 'profile_top_fv' and 'profile_right_fv'
    """
    # Top (horizontal) profile
    start_top = (0, 32)  # (x,y) coordinates
    end_top = (64, 32)
    profile_top = profile_line(processed_image, start_top, end_top, linewidth=5)

    # Right (vertical) profile
    start_right = (32, 0)  # (x,y) coordinates
    end_right = (32, 64)
    profile_right = profile_line(processed_image, start_right, end_right, linewidth=5)

    df.at[i,'profile_top_fv'] = profile_top
    df.at[i,'profile_right_fv'] = profile_right


def plot_intensity(processed_image, profile, start, end):
    """
    Visualizes the intensity profile along a line in the image.
    
    Args:
        processed_image (numpy.ndarray): The image being analyzed
        profile (numpy.ndarray): The intensity profile values
        start (tuple): Starting (x,y) coordinates of profile line
        end (tuple): Ending (x,y) coordinates of profile line
        
    Returns:
        None. Displays a matplotlib plot showing the image and profile.
    """
    fig, ax = plt.subplots(2, 1, figsize=(15, 9))
    
    # Plot image with profile line
    ax[0].imshow(processed_image, cmap=plt.cm.gist_earth, interpolation='gaussian', alpha=1)
    ax[0].plot([start[0], end[0]], [start[1], end[1]], 'r-', lw=3)
    
    # Plot intensity profile
    ax[1].plot(profile)
    ax[1].set_title(f'Profile data points = {profile.shape[0]}')
    
    plt.tight_layout()
    plt.show()

