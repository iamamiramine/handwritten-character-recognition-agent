# Character Recognition System

A machine learning-based Optical Character Recognition (OCR) system that processes individual character images using multiple feature extraction techniques and Support Vector Machine (SVM) classification.

## Overview

This system implements a complete OCR pipeline including:
- Image preprocessing and normalization
- Multiple feature extraction methods
- SVM-based classification
- Model evaluation and prediction

## Features

### Feature Extraction Methods
- **HOG (Histogram of Oriented Gradients)**
  - Captures local shape information
  - Uses 9 orientations, 8x8 pixels per cell
  - 2x2 cells per block with L2 normalization

- **Zoning**
  - Divides image into 16x16 pixel zones
  - Counts white pixels in each zone
  - Creates density-based feature vector

- **Profile Analysis**
  - Takes horizontal and vertical intensity profiles
  - Uses 5-pixel line width
  - Captures character shape along central axes

- **Edge Detection**
  - Uses Prewitt operators
  - Detects horizontal and vertical edges
  - Captures character boundaries

- **Projection Profiles**
  - Calculates horizontal and vertical projections
  - Includes both sum and mean projections
  - Captures character density distribution

- **SIFT (Experimental)**
  - Scale and rotation invariant features
  - Currently in experimental phase

### Preprocessing Pipeline
1. Grayscale conversion
2. Binary thresholding
3. Contour detection and cropping
4. Size normalization (64x64)
5. Skeletonization

## Installation
1. Clone the repository:

```bash
git clone https://github.com/yourusername/character-recognition.git
cd character-recognition
```

2. Install required packages:
```bash
pip install numpy pandas scikit-image scikit-learn opencv-python matplotlib
```

## Project Structure
.
├── main.py # Main entry point
├── scripts/
│ ├── Run.py # Pipeline orchestration
│ ├── Pre_Processing.py # Image preprocessing
│ ├── Classification.py # SVM model training/prediction
│ ├── Projection.py # Projection features
│ └── features/
│ ├── HOG.py # HOG feature extraction
│ ├── Zoning.py # Zoning feature extraction
│ ├── Profile.py # Profile feature extraction
│ ├── Edges.py # Edge feature extraction
│ └── SIFT.py # SIFT feature extraction


## Usage

The system can be run using the command-line interface:

```bash
python main.py --input <input_dir> \
--labels <labels_file> \
--output <output_dir> \
--mode <train|eval|predict> \
--features <feature_list>
```


### Arguments
- `--input`: Directory containing input images
- `--labels`: CSV file with image labels
- `--output`: Directory for output files
- `--mode`: Operation mode (train/eval/predict)
- `--features`: List of features to extract (default: hog zone)

### Examples

Training with multiple features:
```bash
python main.py --input ./Data/Training_Data/data_set_1_training_data/ \
--labels ./Data/Training_Data/data_set_1_training_data.csv \
--output ./Output \
--mode train \
--features hog zone profile
```

Model evaluation:
```bash
python main.py --input ./Data/Training_Data/data_set_1_training_data/ \
--labels ./Data/Training_Data/data_set_1_training_data.csv \
--output ./Output \
--mode eval
```

Character prediction:
```bash
python main.py --input ./Data/Inputs/Img/ \
--labels ./Data/Inputs/inputs.csv \
--output ./Output \
--mode predict \
--features hog zone edges
```


## Data Format

### Required Directory Structure

Data/
├── Training_Data/
│ ├── data_set_1_training_data/
│ │ ├── char1.png
│ │ ├── char2.png
│ │ └── ...
│ └── data_set_1_training_data.csv
└── Inputs/
├── Img/
│ ├── test1.png
│ ├── test2.png
│ └── ...
└── inputs.csv


### Label File Format (CSV)

```csv
filename,label
char1.png,A
char2.png,B
...
```


## Model Details

### SVM Classifier
- Uses scikit-learn SVM implementation
- Feature preprocessing:
  1. MinMaxScaler normalization
  2. StandardScaler standardization
- 80-20 train-test split
- Random state: 0 (for reproducibility)

### Output Files
- Trained model: `Model/ensemble.joblib`
- Feature vectors: CSV and JSON formats
- Classification reports and accuracy metrics

## Performance Metrics

The system provides:
- Accuracy score
- Classification report (precision, recall, F1-score)
- Per-class accuracy statistics

## Limitations

- Fixed input image size (64x64)
- Single character recognition only
- Limited to predefined character classes
- SIFT features not fully implemented

## Future Improvements

1. Support for variable image sizes
2. Additional feature extraction methods
3. Hyperparameter optimization
4. Multi-character recognition
5. Deep learning integration
6. Real-time processing support

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
