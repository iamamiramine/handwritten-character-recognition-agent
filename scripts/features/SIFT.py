from skimage.feature import match_descriptors, plot_matches, SIFT

"""
This module extracts Scale-Invariant Feature Transform (SIFT) features from character images.
SIFT features are scale and rotation invariant descriptors of local image regions.
Note: Currently this module is experimental and features are not being used in classification.
"""

def SIFT_Training(processed_image, df, i):
    """
    Extracts SIFT features from training images.
    Currently only prints descriptors without storing them.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image
        df (pandas.DataFrame): DataFrame to store features (not used currently)
        i (int): Index of current image in DataFrame
        
    Returns:
        None. Currently only prints descriptors.
    
    TODO:
        - Store descriptors in DataFrame instead of printing
        - Determine how to use variable-length SIFT features for classification
    """
    descriptor_extractor = SIFT()
    descriptor_extractor.detect_and_extract(processed_image)
    descriptors = descriptor_extractor.descriptors

    print(descriptors)


def SIFT_Input(processed_image, df, i):
    """
    Extracts SIFT features from input images.
    Currently only prints descriptors without storing them.
    
    Args:
        processed_image (numpy.ndarray): Pre-processed binary image
        df (pandas.DataFrame): DataFrame to store features (not used currently)
        i (int): Index of current image in DataFrame
        
    Returns:
        None. Currently only prints descriptors.
    
    TODO:
        - Store descriptors in DataFrame instead of printing
        - Determine how to use variable-length SIFT features for classification
    """
    descriptor_extractor = SIFT()
    descriptor_extractor.detect_and_extract(processed_image)
    descriptors = descriptor_extractor.descriptors

    print(descriptors)