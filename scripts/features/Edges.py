import numpy as np
from skimage.filters import prewitt_h,prewitt_v

"""
This module extracts edge features from character images using Prewitt filters.
Prewitt operators detect vertical and horizontal edges in the image.
"""

def Edges_Training(processed_image, df, i):
    """
    Extracts edge features from training images using Prewitt operators.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image
        df (pandas.DataFrame): DataFrame to store the features
        i (int): Index of current image in DataFrame
        
    Returns:
        None. Features are stored in DataFrame columns 'edges_horizontal_fv' and 'edges_vertical_fv'
    """
    # Calculate horizontal edges using prewitt kernel
    edges_prewitt_horizontal = prewitt_h(processed_image)
    # Calculate vertical edges using prewitt kernel
    edges_prewitt_vertical = prewitt_v(processed_image)

    # Store features in DataFrame
    df.at[i,'edges_horizontal_fv'] = edges_prewitt_horizontal
    df.at[i,'edges_vertical_fv'] = edges_prewitt_vertical


def Edges_Input(processed_image, df, i):
    """
    Extracts edge features from input images using Prewitt operators.
    Uses same process as training to maintain consistency.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image
        df (pandas.DataFrame): DataFrame to store the features
        i (int): Index of current image in DataFrame
        
    Returns:
        None. Features are stored in DataFrame columns 'edges_horizontal_fv' and 'edges_vertical_fv'
    """
    # Calculate horizontal edges using prewitt kernel
    edges_prewitt_horizontal = prewitt_h(processed_image)
    # Calculate vertical edges using prewitt kernel
    edges_prewitt_vertical = prewitt_v(processed_image)

    df.at[i,'edges_horizontal_fv'] = edges_prewitt_horizontal
    df.at[i,'edges_vertical_fv'] = edges_prewitt_vertical