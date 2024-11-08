import cv2
import numpy as np

"""
This module extracts projection features from character images.
Projection profiles capture the distribution of pixel intensities along 
horizontal and vertical axes, providing shape information about the character.
"""

def Hist_Proj_Training(processed_image, df, i):
    """
    Extracts projection features from training images.
    Calculates both sum and mean projections along horizontal and vertical axes.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image (64x64)
        df (pandas.DataFrame): DataFrame to store the features
        i (int): Index of current image in DataFrame
        
    Returns:
        None. Features are stored in DataFrame columns:
            - 'horizontal_proj_fv': Sum of pixels along rows
            - 'mean_horizontal_proj_fv': Mean of pixels along rows
            - 'vertical_proj_fv': Sum of pixels along columns
            - 'mean_vertical_proj_fv': Mean of pixels along columns
    """
    # Horizontal projections (along rows)
    horizontal_projection = np.sum(processed_image, axis=1)
    mean_horizontal_projection = np.mean(processed_image, axis=1)

    # Vertical projections (along columns)
    vertical_projection = np.sum(processed_image, axis=0)
    mean_vertical_projection = np.mean(processed_image, axis=0)

    # Store features in DataFrame
    df.at[i,'horizontal_proj_fv'] = horizontal_projection
    df.at[i,'mean_horizontal_proj_fv'] = mean_horizontal_projection
    df.at[i,'vertical_proj_fv'] = vertical_projection
    df.at[i,'mean_vertical_proj_fv'] = mean_vertical_projection

def Hist_Proj_Input(processed_image, df, i):
    """
    Extracts projection features from input images.
    Uses same process as training to maintain consistency.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image (64x64)
        df (pandas.DataFrame): DataFrame to store the features
        i (int): Index of current image in DataFrame
        
    Returns:
        None. Features are stored in DataFrame columns:
            - 'horizontal_proj_fv': Sum of pixels along rows
            - 'mean_horizontal_proj_fv': Mean of pixels along rows
            - 'vertical_proj_fv': Sum of pixels along columns
            - 'mean_vertical_proj_fv': Mean of pixels along columns
    """
    # Horizontal projections (along rows)
    horizontal_projection = np.sum(processed_image, axis=1)
    mean_horizontal_projection = np.mean(processed_image, axis=1)

    # Vertical projections (along columns)
    vertical_projection = np.sum(processed_image, axis=0)
    mean_vertical_projection = np.mean(processed_image, axis=0)

    # Store features in DataFrame
    df.at[i,'horizontal_proj_fv'] = horizontal_projection
    df.at[i,'mean_horizontal_proj_fv'] = mean_horizontal_projection
    df.at[i,'vertical_proj_fv'] = vertical_projection
    df.at[i,'mean_vertical_proj_fv'] = mean_vertical_projection
 