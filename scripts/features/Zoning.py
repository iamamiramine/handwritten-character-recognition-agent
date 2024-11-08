import numpy as np

"""
This module implements zoning feature extraction for character images.
Zoning divides the image into equal-sized zones and counts white pixels in each zone
to capture character shape and density information.
"""

def blockshaped(processed_image, nrows, ncols):
    """
    Reshapes a 2D image array into a list of zone blocks.
    
    Args:
        processed_image (numpy.ndarray): Input binary image
        nrows (int): Number of rows per zone
        ncols (int): Number of columns per zone
        
    Returns:
        numpy.ndarray: Reshaped array where each element is a zone block
        
    Raises:
        AssertionError: If image dimensions are not evenly divisible by zone size
        
    Note:
        Zones are ordered from left to right, top to bottom:
        Top Left - Top Right - Middle Left - Middle Right - Bottom Left - Bottom Right
    """
    h, w = processed_image.shape
    assert h % nrows == 0, f"{h} rows is not evenly divisible by {nrows}"
    assert w % ncols == 0, f"{w} cols is not evenly divisible by {ncols}"
    return (processed_image.reshape(h//nrows, nrows, -1, ncols)
            .swapaxes(1,2)
            .reshape(-1, nrows, ncols))

def Zoning_Training(processed_image, df, i):
    """
    Extracts zoning features from training images.
    Divides image into 16x16 pixel zones and counts white pixels in each zone.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image (64x64)
        df (pandas.DataFrame): DataFrame to store the features
        i (int): Index of current image in DataFrame
        
    Returns:
        None. Features are stored in DataFrame column 'zone_fv'
    """
    zoned = blockshaped(processed_image, 16, 16)  # Creates 4x4 zones
    zone_fv = []
    for zone in zoned:
        white_pixels = np.sum(zone == 1)  # Count white pixels in zone
        zone_fv.append(white_pixels)

    df.at[i,'zone_fv'] = zone_fv

def Zoning_Input(processed_image, df, i):
    """
    Extracts zoning features from input images.
    Uses same parameters as training to maintain consistency.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image (64x64)
        df (pandas.DataFrame): DataFrame to store the features
        i (int): Index of current image in DataFrame
        
    Returns:
        None. Features are stored in DataFrame column 'zone_fv'
    """
    zoned = blockshaped(processed_image, 16, 16)  # Creates 4x4 zones
    zone_fv = []
    for zone in zoned:
        white_pixels = np.sum(zone == 1)  # Count white pixels in zone
        zone_fv.append(white_pixels)

    df.at[i,'zone_fv'] = zone_fv