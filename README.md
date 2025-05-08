# neonates-seizure-detection
Reproducing Attention-Based Network for Weak Labels in Neonatal Seizure Detection


Project Overview
This implementation provides a complete pipeline for:

Data Processing: Reading and preprocessing EEG data (.edf files) and annotation files
Model Training: Training a deep learning model based on the Inception architecture
Visualization: Generating insights through visualizations of model performance

Directory Structure
Organize your data in the following structure:
project/
├── data/
│   ├── eeg1.edf, eeg2.edf, ...
│   ├── annotations_2017_A.csv
│   ├── annotations_2017_B.csv
│   ├── annotations_2017_C.csv
│   ├── clinical_information.csv
│   └── processed/  (will be created automatically)
├── models/  (will be created automatically)
├── results/  (will be created automatically)
├── visualizations/  (will be created automatically)
├── scripts/
│   ├── model.py
│   ├── eeg_data_processor.py
│   ├── eeg_model_training.py
│   └── eeg_visualization.py
└── pipeline_runner.py
|__ run-batch-processing.sh

Installation Requirements
Create a Python environment with the following dependencies:
bashpip install numpy pandas matplotlib seaborn scikit-learn tensorflow mne
Step 1: Setting Up the Model
The model.py file defines the architecture using TensorFlow/Keras. The key components are:

Inception1D architecture: A CNN-based model for processing time-series data
Feature extraction: Processes each EEG channel independently
Classification: Final layers to classify seizure vs. non-seizure

The model_DL2 function is the main entry point that creates the model.
Step 2: Data Processing
The data processor performs these steps:

Loads EEG data (.edf files) using MNE
Processes annotation files
Windows the data for model input
Normalizes the features
Creates train/val/test splits

To run the data processing:
bashpython pipeline_runner.py
Step 3: Model Training
The training script:

Loads the processed data
Creates the model architecture
Compiles with appropriate loss function
Trains with early stopping and learning rate scheduling
Evaluates on test data
Saves performance metrics

To run just the training step:
bashpython pipeline_runner.py --skip-preprocessing
Step 4: Visualization
The visualization script creates:

Prediction distributions: How seizure probabilities are distributed
Channel importance: Relative importance of EEG channels
Example predictions: Visualizations of correct/incorrect predictions
Performance metrics: Confusion matrix, ROC curve, and precision-recall curve

Batch Processing for Large Datasets
For large EEG datasets, processing all patients at once can lead to memory issues. The batch processing approach solves this:
project/scripts/batch_eeg_processor.py
This script:

Processes patients in small batches (e.g., 5 patients at a time)
Saves individual patient data to separate folders
Combines data only after all patients are processed
Uses memory-efficient operations to prevent crashes

To run batch processing:
bash# Process patients 1-5
python project/scripts/batch_eeg_processor.py 1 5

# Process patients 6-10
python project/scripts/batch_eeg_processor.py 6 10

# Combine all processed patient data
python project/scripts/batch_eeg_processor.py
Alternatively, use the batch runner script:
bash./run-batch-processing.sh
This automatically processes all patients in small batches and combines the results.
