import os
import skimage
from skimage import morphology, data, io, measure, data, exposure, img_as_bool
from skimage.color import rgb2gray
from skimage.transform import rescale, rotate, resize
from skimage.filters import threshold_otsu, threshold_local
from skimage.feature import corner_harris, corner_peaks, canny, hog
from skimage.draw import ellipse
from skimage.measure import label, regionprops, regionprops_table, moments
from skimage.util import crop
import cv2
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
This module handles image pre-processing for OCR character recognition.
The preprocessing pipeline includes:
1. Grayscale conversion
2. Binary thresholding
3. Contour detection and cropping
4. Resizing to standard dimensions
5. Skeletonization
"""

# Global variable for skeleton intensity values
skeleton_int = np.zeros((400, 400), dtype=int)

def show_image(image, title='Image', cmap_type='gray'):
    """
    Displays an image using matplotlib.
    
    Args:
        image (numpy.ndarray): Image to display
        title (str): Title for the plot
        cmap_type (str): Colormap type for display
        
    Returns:
        None. Displays the plot.
    """
    plt.imshow(image, cmap=cmap_type)
    plt.title(title)
    plt.axis('off')

def show_image_contour(image, title='Image'):
    """
    Displays an image with its bounding box contour.
    
    Args:
        image (numpy.ndarray): Image to display
        title (str): Title for the plot
        
    Returns:
        None. Displays and saves the plot.
    """
    plt.imshow(image, cmap='gray')
    
    # Draw bounding box
    x1, y1 = 0, 0
    x2 = image.shape[1]
    y2 = image.shape[0]
    bx = (x1, x2, x2, x1, x1)
    by = (y1, y1, y2, y2, y1)
    plt.plot(bx, by, '-b', linewidth=1)

    plt.title(title)
    plt.imsave('filename.png', image, cmap='gray')
    plt.axis('off')
    plt.show()

def image_processing(image):
    """
    Main preprocessing function that converts input images to standardized binary skeletons.
    
    Args:
        image (numpy.ndarray): Input BGR image
        
    Returns:
        numpy.ndarray: Preprocessed binary skeleton image (64x64)
        
    Process:
        1. Convert to grayscale
        2. Binary threshold with inverse
        3. Find and extract character contour
        4. Resize to standard size
        5. Create skeleton
    """
    # Convert to grayscale
    grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Binary threshold
    ret, binaryImage = cv2.threshold(grayscaleImage, 127, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    contours_poly = [None] * len(contours)
   
    # Get bounding rectangles for outer contours
    boundRect = []
    for i, c in enumerate(contours):
        if hierarchy[0][i][3] == -1:  # Only process outer contours
            contours_poly[i] = cv2.approxPolyDP(c, 3, True)
            boundRect.append(cv2.boundingRect(contours_poly[i]))

    # Crop to character bounds
    for i in range(len(boundRect)):
        x, y, w, h = boundRect[i]
        croppedImg = binaryImage[y:y + h, x:x + w]

    # Resize to standard size
    resized = img_as_bool(resize(croppedImg, (500, 500)))

    # Scale down to 64x64
    binary_image = rescale(resized, 0.128)

    # Create skeleton
    skeleton = morphology.skeletonize(binary_image)
    
    return skeleton

