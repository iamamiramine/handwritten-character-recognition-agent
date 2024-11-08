import numpy as np
import json
from matplotlib import pyplot as plt
from skimage import color
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold, SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from sklearn.manifold import Isomap
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.decomposition import PCA
import pandas as pd
from joblib import dump, load

"""
This module handles the training and prediction of the OCR model using SVM classification.
It processes HOG and Zoning feature vectors extracted from character images to train
a Support Vector Machine classifier for character recognition.

The workflow consists of:
1. Feature normalization using MinMaxScaler
2. Feature standardization using StandardScaler
3. Training SVM classifier
4. Model evaluation and persistence
"""

def Class_eval(df): 
    """
    Trains and evaluates an SVM classifier on the provided feature vectors.
    
    Args:
        df (pandas.DataFrame): DataFrame containing 'hog_fv', 'zone_fv' and 'label' columns
        
    Returns:
        None. Prints classification metrics and saves the trained model.
        
    Notes:
        - Features are normalized using MinMaxScaler then standardized using StandardScaler
        - Uses 80-20 train-test split with random_state=0 for reproducibility
        - Saves trained model to '.\Model\ensemble.joblib'
    """
    # Initialize normalizers
    norm_model = MinMaxScaler()

    # Extract feature vectors from DataFrame
    feature_arr_3 = np.array([df['hog_fv'].iloc[i] for i in range(len(df['hog_fv']))])
    feature_arr_4 = np.array([df['zone_fv'].iloc[i] for i in range(len(df['zone_fv']))])
    y = np.array(df['label']).reshape(len(df['label']), 1)

    # Normalize features
    data_norm_3 = norm_model.fit_transform(feature_arr_3)
    data_norm_4 = norm_model.fit_transform(feature_arr_4)

    # Combine features and labels
    data_frame = np.column_stack((data_norm_3, data_norm_4, y))
    np.random.shuffle(data_frame)

    # Split features and labels
    X = data_frame[:,:-1]
    y = data_frame[:,-1:].ravel()

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Initialize and train SVM
    clf_SVM = svm.SVC()
    clf_SVM.fit(X_train, y_train)

    # Make predictions and evaluate
    y_pred = clf_SVM.predict(X_test)

    # Print metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print('\n')
    print(classification_report(y_test, y_pred))

    correct_labels = count_correct_labels(y_pred, y_test)
    print_stats(correct_labels, len(y_test))

    # Save model
    dump(clf_SVM, '.\Model\ensemble.joblib')


def Class_predict(df): 
    """
    Uses trained SVM model to predict characters from input feature vectors.
    
    Args:
        df (pandas.DataFrame): DataFrame containing 'hog_fv', 'zone_fv' and 'label' columns
        
    Returns:
        None. Prints prediction accuracy and classification report.
        
    Notes:
        - Uses same normalization process as training
        - Loads model from '.\Model\ensemble.joblib'
    """
    # Load trained model
    clf = load('.\Model\ensemble.joblib') 
    norm_model = MinMaxScaler()

    # Extract and normalize features
    feature_arr_3 = np.array([df['hog_fv'].iloc[i] for i in range(len(df['hog_fv']))])
    feature_arr_4 = np.array([df['zone_fv'].iloc[i] for i in range(len(df['zone_fv']))])
    y = np.array(df['label']).reshape(len(df['label']), 1)

    data_norm_3 = norm_model.fit_transform(feature_arr_3) 
    data_norm_4 = norm_model.fit_transform(feature_arr_4)

    # Combine features and labels
    data_frame = np.column_stack((data_norm_3, data_norm_4, y))
    
    # Split features and true labels
    y_true = data_frame[:,-1]
    X = data_frame[:,:-1]

    # Make predictions
    y_pred = clf.predict(X)

    # Print metrics
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print('\n')
    print(classification_report(y_true, y_pred))

    correct_labels = count_correct_labels(y_pred, y_true)
    print_stats(correct_labels, len(y_true))


def count_correct_labels(predicted_labels, test_labels):
    """
    Counts number of correct predictions.
    
    Args:
        predicted_labels (numpy.ndarray): Predicted character labels
        test_labels (numpy.ndarray): True character labels
        
    Returns:
        int: Number of correct predictions
    """
    correct = 0
    for p, t in zip(predicted_labels, test_labels):
        if p[0] == t:
            correct += 1
    return correct


def print_stats(correct_labels, len_test_labels):
    """
    Prints accuracy statistics.
    
    Args:
        correct_labels (int): Number of correct predictions
        len_test_labels (int): Total number of test samples
        
    Returns:
        None. Prints accuracy metrics.
    """
    print("Correct labels =", correct_labels)
    print("Accuracy =", (correct_labels * 100 / float(len_test_labels)))