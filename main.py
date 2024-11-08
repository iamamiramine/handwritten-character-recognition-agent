import argparse
import os
import sys
from Run import run_for_training, run_for_inputs, run_eval, run_predict

"""
Main entry point for the OCR system.
Handles command line arguments and orchestrates the complete pipeline.
"""

def parse_args():
    """
    Parses command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='OCR System for Character Recognition')
    
    # Required arguments
    parser.add_argument('--input', type=str, required=True,
                      help='Path to input dataset directory containing images')
    parser.add_argument('--labels', type=str, required=True,
                      help='Path to CSV file containing labels')
    parser.add_argument('--output', type=str, required=True,
                      help='Path to output directory')
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True, 
                      choices=['train', 'eval', 'predict'],
                      help='Operation mode: train, eval, or predict')
    
    # Optional arguments
    parser.add_argument('--features', type=str, nargs='+',
                      choices=['hog', 'zone', 'profile', 'edges', 'projection', 'sift'],
                      default=['hog', 'zone'],
                      help='Feature types to extract (default: hog zone)')
    
    return parser.parse_args()

def setup_directories(args):
    """
    Creates necessary directories and validates paths.
    
    Args:
        args: Parsed command line arguments
        
    Raises:
        FileNotFoundError: If input paths don't exist
    """
    # Check input paths
    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input directory not found: {args.input}")
    if not os.path.exists(args.labels):
        raise FileNotFoundError(f"Labels file not found: {args.labels}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, 'Model'), exist_ok=True)

def main():
    """
    Main function that runs the OCR pipeline based on command line arguments.
    """
    # Parse arguments
    args = parse_args()
    
    try:
        # Setup directories
        setup_directories(args)
        
        # Set up environment variables for paths
        os.environ['OCR_INPUT_DIR'] = args.input
        os.environ['OCR_LABELS_FILE'] = args.labels
        os.environ['OCR_OUTPUT_DIR'] = args.output
        
        # Execute requested operation
        if args.mode == 'train':
            print(f"Training model with features: {', '.join(args.features)}")
            run_for_training(features=args.features)
            print("Training complete. Model saved.")
            
        elif args.mode == 'eval':
            print("Evaluating model...")
            run_eval()
            print("Evaluation complete.")
            
        elif args.mode == 'predict':
            print(f"Running prediction with features: {', '.join(args.features)}")
            run_for_inputs(features=args.features)
            run_predict()
            print("Prediction complete.")
            
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 