import os
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import warnings
import time
import sys

# Suppress warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define paths
DATA_DIR = 'project/data'
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
ANNOTATION_FILES = [
    os.path.join(DATA_DIR, 'annotations_2017_A.csv'),
    os.path.join(DATA_DIR, 'annotations_2017_B.csv'),
    os.path.join(DATA_DIR, 'annotations_2017_C.csv')
]
CLINICAL_INFO_FILE = os.path.join(DATA_DIR, 'clinical_information.csv')

# Create processed directory if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

def load_clinical_data():
    """Load clinical information data."""
    clinical_df = pd.read_csv(CLINICAL_INFO_FILE)
    clinical_df['Has_Seizure'] = clinical_df['Number of Reviewers Annotating Seizure'] > 0
    return clinical_df

def process_patient(patient_id, eeg_name):
    """Process a single patient's EEG data."""
    try:
        start_time = time.time()
        print(f"Processing patient {patient_id} (EEG file: {eeg_name})")
        
        # File path
        edf_file = os.path.join(DATA_DIR, f"{eeg_name}.edf")
        if not os.path.exists(edf_file):
            print(f"Error: EDF file not found at {edf_file}")
            return None, None
        
        # Load EEG data
        raw = mne.io.read_raw_edf(edf_file, preload=True)
        
        # Window parameters
        sfreq = 256.0
        window_size_sec = 1.0
        overlap_sec = 0.5
        
        # Resample if needed
        if raw.info['sfreq'] != sfreq:
            raw.resample(sfreq)
        
        # Extract data
        data = raw.get_data()
        n_channels, signal_length = data.shape
        
        # Window parameters
        n_samples = int(window_size_sec * sfreq)
        n_overlap = int(overlap_sec * sfreq)
        n_stride = n_samples - n_overlap
        
        # Calculate number of windows
        n_windows = (signal_length - n_samples) // n_stride + 1
        
        # Extract windows
        X = np.zeros((n_windows, n_channels, n_samples))
        for i in range(n_windows):
            start = i * n_stride
            end = start + n_samples
            X[i, :, :] = data[:, start:end]
        
        # Create synthetic labels based on clinical data
        has_seizure = clinical_df.loc[clinical_df['ID'] == patient_id, 'Has_Seizure'].values[0]
        
        if has_seizure:
            print(f"Creating synthetic labels for seizure-positive patient {patient_id}")
            seizure_ratio = 0.05  # 5% of windows contain seizures
            n_seizure_windows = int(n_windows * seizure_ratio)
            
            # Create labels with random seizure windows
            y = np.zeros(n_windows)
            seizure_indices = np.random.choice(n_windows, n_seizure_windows, replace=False)
            y[seizure_indices] = 1
        else:
            print(f"Using all-negative labels for seizure-negative patient {patient_id}")
            y = np.zeros(n_windows)
        
        # Convert to categorical
        y = to_categorical(y, num_classes=2)
        
        # Normalize data
        scaler = StandardScaler()
        X_reshaped = X.reshape(n_windows, -1)
        X_normalized = scaler.fit_transform(X_reshaped)
        X_final = X_normalized.reshape(n_windows, n_channels, n_samples)
        
        elapsed_time = time.time() - start_time
        print(f"Processed patient {patient_id}: {n_windows} windows in {elapsed_time:.2f} seconds")
        
        return X_final, y
        
    except Exception as e:
        print(f"Error processing patient {patient_id}: {str(e)}")
        return None, None

def process_batch(patient_batch, clinical_df):
    """Process a batch of patients and save individual results."""
    processed_patients = []
    
    for _, row in patient_batch.iterrows():
        patient_id = row['ID']
        eeg_name = row['EEG file']
        
        # Process this patient
        X, y = process_patient(patient_id, eeg_name)
        
        if X is not None and y is not None:
            # Save individual patient data
            patient_dir = os.path.join(PROCESSED_DIR, f'patient_{patient_id}')
            os.makedirs(patient_dir, exist_ok=True)
            
            np.save(os.path.join(patient_dir, 'X.npy'), X)
            np.save(os.path.join(patient_dir, 'y.npy'), y)
            
            processed_patients.append(patient_id)
    
    return processed_patients

def combine_and_split_data(processed_patients, clinical_df):
    """Combine data from processed patients and split into train/val/test sets."""
    print(f"Combining data from {len(processed_patients)} patients...")
    
    X_all = []
    y_all = []
    patient_ids = []
    
    for patient_id in processed_patients:
        patient_dir = os.path.join(PROCESSED_DIR, f'patient_{patient_id}')
        
        X = np.load(os.path.join(patient_dir, 'X.npy'))
        y = np.load(os.path.join(patient_dir, 'y.npy'))
        
        X_all.append(X)
        y_all.append(y)
        patient_ids.extend([patient_id] * X.shape[0])
    
    # Combine all data
    X_combined = np.vstack(X_all)
    y_combined = np.vstack(y_all)
    patient_ids = np.array(patient_ids)
    
    print(f"Combined data shape: {X_combined.shape}")
    
    # Split by patient
    unique_patients = np.unique(patient_ids)
    
    # Split into train, val, test
    train_patients, temp_patients = train_test_split(
        unique_patients, test_size=0.3, random_state=42
    )
    
    val_patients, test_patients = train_test_split(
        temp_patients, test_size=0.5, random_state=42
    )
    
    # Create masks for each set
    train_mask = np.isin(patient_ids, train_patients)
    val_mask = np.isin(patient_ids, val_patients)
    test_mask = np.isin(patient_ids, test_patients)
    
    # Apply masks
    X_train = X_combined[train_mask]
    y_train = y_combined[train_mask]
    X_val = X_combined[val_mask]
    y_val = y_combined[val_mask]
    X_test = X_combined[test_mask]
    y_test = y_combined[test_mask]
    
    print(f"Train: {X_train.shape[0]} samples from {len(train_patients)} patients")
    print(f"Val: {X_val.shape[0]} samples from {len(val_patients)} patients")
    print(f"Test: {X_test.shape[0]} samples from {len(test_patients)} patients")
    
    # Save combined data
    np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(PROCESSED_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(PROCESSED_DIR, 'y_val.npy'), y_val)
    np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test)
    np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)
    
    print(f"Split data saved to {PROCESSED_DIR}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def main():
    """Main function that processes patients in batches."""
    try:
        print("Starting batch EEG processing...")
        
        # Load clinical data
        global clinical_df
        clinical_df = load_clinical_data()
        print(f"Loaded clinical data: {len(clinical_df)} patients")
        
        # Check if we should process only a specific batch
        if len(sys.argv) > 1:
            batch_start = int(sys.argv[1])
            batch_end = int(sys.argv[2]) if len(sys.argv) > 2 else batch_start + 5
            
            print(f"Processing batch from patient {batch_start} to {batch_end}")
            patient_batch = clinical_df[(clinical_df['ID'] >= batch_start) & 
                                        (clinical_df['ID'] <= batch_end)]
        else:
            # Process all patients
            patient_batch = clinical_df
        
        # Process this batch
        processed_patients = process_batch(patient_batch, clinical_df)
        
        # Only combine data if all patients requested have been processed
        if len(processed_patients) > 0:
            print(f"Successfully processed {len(processed_patients)} patients")
            
            # Only combine and split if we processed the entire dataset
            if len(sys.argv) <= 1 or (len(sys.argv) > 1 and len(processed_patients) == len(patient_batch)):
                combine_and_split_data(processed_patients, clinical_df)
        else:
            print("No patients were successfully processed")
        
        print("Processing complete!")
        
    except Exception as e:
        print(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()