import os
import cv2
import csv
import pandas as pd
from skimage.color import rgb2gray

from Pre_Processing import image_processing
from Classification import Class_eval, Class_predict
from features.HOG import HOG_Training, HOG_Input
from features.Zoning import Zoning_Training, Zoning_Input
from features.Profile import Profile_Training, Profile_Input
from features.SIFT import SIFT_Training, SIFT_Input
from features.Edges import Edges_Training, Edges_Input
from Projection import Hist_Proj_Training, Hist_Proj_Input

"""
Main orchestration module for the OCR system.
This module coordinates the entire OCR pipeline including:
1. Data loading and preprocessing
2. Feature extraction
3. Model training and evaluation
4. Character prediction

The system supports both batch processing of training data
and individual character processing for testing.
"""

# Feature extraction options
FEATURE_EXTRACTORS = {
    'hog': (HOG_Training, HOG_Input),
    'zone': (Zoning_Training, Zoning_Input),
    'profile': (Profile_Training, Profile_Input),
    'edges': (Edges_Training, Edges_Input),
    'projection': (Hist_Proj_Training, Hist_Proj_Input),
    'sift': (SIFT_Training, SIFT_Input)
}

def run_for_training(features=None):
    """
    Processes the training dataset through the OCR pipeline.
    
    Args:
        features (list): List of feature names to extract. If None, uses default ['hog', 'zone']
    
    Process:
        1. Loads character images and labels from training directory
        2. Preprocesses each image (binarization, normalization)
        3. Extracts specified features
        4. Saves processed features to CSV/JSON for model training
        
    Notes:
        Training data should be in './Data/Training_Data/data_set_1_training_data/'
        Labels should be in './Data/Training_Data/data_set_1_training_data.csv'
    """
    if features is None:
        features = ['hog', 'zone']  # Default features
    
    # Validate feature names
    for feature in features:
        if feature not in FEATURE_EXTRACTORS:
            raise ValueError(f"Unknown feature type: {feature}. Available features: {list(FEATURE_EXTRACTORS.keys())}")

    # Load training data
    df = pd.read_csv("./Data/Training_Data/data_set_1_training_data.csv")
    df = df.astype({'label': 'string'})

    # Process each image
    for i, filename in enumerate(os.listdir('./Data/Training_Data/data_set_1_training_data/')):
        # Load and preprocess image
        image = cv2.imread(os.path.join('./Data/Training_Data/data_set_1_training_data/', filename))
        processed_image = image_processing(image)

        # Extract selected features
        for feature in features:
            training_func = FEATURE_EXTRACTORS[feature][0]  # Get training function
            training_func(processed_image, df, i)

    # Save features
    df.to_csv('./Data/Training_Data/data_set_1_training_data.csv', index=False)
    df.to_json('./Data/Training_Data/data_set_1_training_data.json')


def run_for_inputs(features=None):
    """
    Processes input images for character prediction.
    
    Args:
        features (list): List of feature names to extract. If None, uses default ['hog', 'zone']
    
    Process:
        1. Loads input images from inputs directory
        2. Preprocesses each image
        3. Extracts specified features
        4. Saves features for prediction
        
    Notes:
        Input images should be in './Data/Inputs/Img/'
        Labels should be in './Data/Inputs/inputs.csv'
    """
    if features is None:
        features = ['hog', 'zone']  # Default features
    
    # Validate feature names
    for feature in features:
        if feature not in FEATURE_EXTRACTORS:
            raise ValueError(f"Unknown feature type: {feature}. Available features: {list(FEATURE_EXTRACTORS.keys())}")

    # Load input data
    df = pd.read_csv("./Data/Inputs/inputs.csv")
    df = df.astype({'label': 'string'})

    # Process each image
    for i, filename in enumerate(os.listdir('./Data/Inputs/Img/')):
        # Load and preprocess image
        image = cv2.imread(os.path.join('./Data/Inputs/Img/', filename))
        processed_image = image_processing(image)

        # Extract selected features
        for feature in features:
            input_func = FEATURE_EXTRACTORS[feature][1]  # Get input function
            input_func(processed_image, df, i)

    # Save features
    df.to_csv('./Data/Inputs/inputs.csv', index=False)
    df.to_json('./Data/Inputs/inputs.json')


def run_individual(features=None):
    """
    Processes a single image for testing/visualization purposes.
    
    Args:
        features (list): List of feature names to extract. If None, uses default ['hog', 'zone']
    
    Notes:
        Test image path is hardcoded to './Letters/null13.png'
    """
    if features is None:
        features = ['hog', 'zone']  # Default features

    image = cv2.imread('./Letters/null13.png')
    processed_image = image_processing(image)
    
    # Extract selected features
    for feature in features:
        if feature not in FEATURE_EXTRACTORS:
            print(f"Warning: Unknown feature type: {feature}")
            continue
        input_func = FEATURE_EXTRACTORS[feature][1]
        input_func(processed_image, None, 0)  # Note: DataFrame is None for visualization


def run_eval():
    """
    Evaluates the OCR model on training data.
    Loads feature data and runs model evaluation.
    
    Notes:
        Uses JSON format to preserve data types
    """
    df = pd.read_json("./Data/Training_Data/data_set_1_training_data.json")
    df = df.astype({'label': 'string'})
    Class_eval(df)


def run_predict():
    """
    Runs prediction on input images using trained model.
    Loads input features and generates predictions.
    
    Notes:
        Uses JSON format to preserve data types
    """
    df = pd.read_json('./Data/Inputs/inputs.json')
    df = df.astype({'label': 'string'})
    Class_predict(df)


# Example usage:
# run_for_training(['hog', 'zone', 'profile'])  # Train with multiple features
# run_for_inputs(['hog', 'zone', 'edges'])      # Process inputs with different features
# run_individual(['hog', 'profile'])            # Test individual image with specific features
# run_eval()                                    # Model Evaluation
# run_predict()                                 # Prediction Execution
