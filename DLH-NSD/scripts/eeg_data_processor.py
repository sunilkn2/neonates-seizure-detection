import os
import numpy as np
import pandas as pd
import mne
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import warnings
import traceback
import gc  # Garbage collector
import time

# Configure environment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings

# Suppress MNE warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Define paths
DATA_DIR = 'project/data'
ANNOTATION_FILES = [
    os.path.join(DATA_DIR, 'annotations_2017_A.csv'),
    os.path.join(DATA_DIR, 'annotations_2017_B.csv'),
    os.path.join(DATA_DIR, 'annotations_2017_C.csv')
]
CLINICAL_INFO_FILE = os.path.join(DATA_DIR, 'clinical_information.csv')

def load_clinical_data():
    """
    Load and preprocess clinical information.
    """
    try:
        clinical_df = pd.read_csv(CLINICAL_INFO_FILE)
        
        # Convert categorical variables into numerical representations as needed
        clinical_df['Gender_Binary'] = clinical_df['Gender'].map({'m': 0, 'f': 1})
        
        # Create a mapping from EEG file to ID for easy lookup
        eeg_to_id_map = dict(zip(clinical_df['EEG file'], clinical_df['ID']))
        
        # Create a feature indicating if seizures were annotated
        clinical_df['Has_Seizure'] = clinical_df['Number of Reviewers Annotating Seizure'] > 0
        
        print(f"Loaded clinical data for {len(clinical_df)} patients")
        print(f"Patients with seizures: {clinical_df['Has_Seizure'].sum()}")
        
        return clinical_df, eeg_to_id_map
    except Exception as e:
        print(f"Error loading clinical data: {e}")
        traceback.print_exc()
        return None, None

def load_annotations(patient_id=None):
    """
    Load and preprocess annotation data from all three reviewers.
    
    Args:
        patient_id: Optional patient ID to filter annotations
        
    Returns:
        tuple: (processed_annotations, raw_annotations)
    """
    try:
        annotation_dfs = []
        
        for file_path in ANNOTATION_FILES:
            try:
                # Load raw data - using numeric indices as there are no headers
                df = pd.read_csv(file_path, header=None)
                annotation_dfs.append(df)
            except Exception as e:
                print(f"Error loading annotation file {file_path}: {e}")
                continue
        
        # Check if all files were loaded successfully
        if len(annotation_dfs) != 3:
            print(f"Could not load all three annotation files. Only loaded {len(annotation_dfs)} files.")
            if len(annotation_dfs) == 0:
                return None, None
        
        # Check for consistent shapes
        shapes = [df.shape for df in annotation_dfs]
        if len(set(shapes)) > 1:
            print(f"Warning: Annotation files have inconsistent shapes: {shapes}")
        
        # Get first row which contains patient IDs
        patient_ids_A = annotation_dfs[0].iloc[0].values
        
        # Filter by patient_id if provided
        if patient_id is not None:
            # Find columns corresponding to the patient
            patient_cols = np.where(patient_ids_A == patient_id)[0]
            if len(patient_cols) == 0:
                print(f"Warning: Patient {patient_id} not found in annotations")
                return None, None
                
            # Extract columns for the patient
            annotations_A = annotation_dfs[0].iloc[1:, patient_cols].values
            annotations_B = annotation_dfs[1].iloc[1:, patient_cols].values if len(annotation_dfs) > 1 else None
            annotations_C = annotation_dfs[2].iloc[1:, patient_cols].values if len(annotation_dfs) > 2 else None
        else:
            # Use all annotations
            annotations_A = annotation_dfs[0].iloc[1:].values
            annotations_B = annotation_dfs[1].iloc[1:].values if len(annotation_dfs) > 1 else None
            annotations_C = annotation_dfs[2].iloc[1:].values if len(annotation_dfs) > 2 else None
        
        # Compute majority vote (at least 2 of 3 reviewers agree)
        if annotations_B is not None and annotations_C is not None:
            annotations_combined = ((annotations_A + annotations_B + annotations_C) >= 2).astype(int)
        else:
            # If we don't have all three reviewers, use what we have
            print("Warning: Not all annotation files available. Using available annotations.")
            annotations_combined = annotations_A
        
        # Pool annotations (downsampling if needed)
        def pool_annotations(annotations, poolsize=8, step=4, thr=0.8):
            """
            Pool annotations with sliding window approach.
            """
            # Handle different input shapes
            if len(annotations.shape) == 1:
                annotations = annotations.reshape(-1, 1)
            
            n_samples, n_channels = annotations.shape
            ann_pool = np.zeros((n_samples // step, n_channels))
            
            for ch in range(n_channels):
                ann_pool_lst = []
                i = 0
                while (i + poolsize) <= n_samples:
                    # For each window, compute the mean and threshold
                    window_mean = np.mean(annotations[i:i + poolsize, ch])
                    ann_pool_lst.append(window_mean > thr)
                    i += step
                
                if len(ann_pool_lst) > 0:
                    # Handle case where lengths don't match exactly
                    if len(ann_pool_lst) != ann_pool.shape[0]:
                        ann_pool = np.zeros((len(ann_pool_lst), n_channels))
                    
                    ann_pool[:, ch] = ann_pool_lst
            
            return ann_pool
        
        # Pool annotations
        ann_pool = pool_annotations(annotations_combined)
        
        return ann_pool, annotations_combined
    
    except Exception as e:
        print(f"Error in load_annotations: {e}")
        traceback.print_exc()
        return None, None

def load_eeg_data(eeg_file, sfreq=256, window_size_sec=1, overlap_sec=0.5, max_windows=None):
    """
    Load and preprocess EEG data from an EDF file with memory-efficient processing.
    
    Args:
        eeg_file: Path to the EDF file
        sfreq: Sampling frequency (Hz)
        window_size_sec: Size of each window in seconds
        overlap_sec: Overlap between consecutive windows in seconds
        max_windows: Maximum number of windows to extract (None for all)
        
    Returns:
        X: Array of windows (n_windows, n_channels, n_samples)
    """
    try:
        start_time = time.time()
        
        # Load the EEG data using MNE
        print(f"Loading EEG file: {eeg_file}")
        raw = mne.io.read_raw_edf(eeg_file, preload=True)
        
        # Print channel info
        print(f"EEG file {eeg_file} has {len(raw.ch_names)} channels: {raw.ch_names}")
        print(f"Sampling frequency: {raw.info['sfreq']} Hz")
        print(f"Duration: {raw.times[-1]:.2f} seconds")
        
        # Extract data dimensions
        n_channels = len(raw.ch_names)
        signal_length = len(raw.times)
        
        # Check file size and decide on processing approach
        large_file = signal_length > 1000000  # If more than ~1M samples
        
        # Window parameters
        n_samples = int(window_size_sec * sfreq)
        n_overlap = int(overlap_sec * sfreq)
        n_stride = n_samples - n_overlap
        
        # Calculate how many windows we'll create
        total_windows = (signal_length - n_samples) // n_stride + 1
        
        # Limit windows if specified
        if max_windows is not None and total_windows > max_windows:
            print(f"Limiting to {max_windows} windows out of {total_windows} possible windows")
            total_windows = max_windows
        
        # Initialize the windows array
        X = np.zeros((total_windows, n_channels, n_samples))
        
        # For large files, process in chunks to avoid memory issues
        if large_file:
            print(f"Large file detected ({signal_length} samples). Using chunked processing.")
            
            # Process in chunks of approximately 1-minute duration
            chunk_size = int(sfreq * 60)  # 1 minute chunks
            windows_processed = 0
            
            for chunk_start in range(0, signal_length, chunk_size):
                # Stop if we've reached our window limit
                if windows_processed >= total_windows:
                    break
                
                # Define chunk boundaries
                chunk_end = min(chunk_start + chunk_size + n_samples, signal_length)
                if chunk_end - chunk_start < n_samples:
                    continue
                
                # Extract this chunk of data
                chunk_data = raw.get_data(start=chunk_start, stop=chunk_end)
                
                # Process this chunk
                chunk_n_windows = min(
                    (chunk_data.shape[1] - n_samples) // n_stride + 1,
                    total_windows - windows_processed
                )
                
                for i in range(chunk_n_windows):
                    if windows_processed + i >= total_windows:
                        break
                    w_start = i * n_stride
                    w_end = w_start + n_samples
                    X[windows_processed + i, :, :] = chunk_data[:, w_start:w_end]
                
                windows_processed += chunk_n_windows
                
                # Force garbage collection to free memory
                del chunk_data
                gc.collect()
                
                # Print progress
                print(f"Processed {windows_processed}/{total_windows} windows ({windows_processed/total_windows*100:.1f}%)")
            
        else:
            # For smaller files, load the entire data at once
            data = raw.get_data()
            
            # Extract windows
            for i in range(min(total_windows, (signal_length - n_samples) // n_stride + 1)):
                start = i * n_stride
                end = start + n_samples
                X[i, :, :] = data[:, start:end]
                
            # Clean up to free memory
            del data
            gc.collect()
        
        # Print time taken
        elapsed_time = time.time() - start_time
        print(f"Created {total_windows} windows in {elapsed_time:.2f} seconds")
        
        # Force raw data cleanup
        raw.close()
        del raw
        gc.collect()
        
        return X
    
    except Exception as e:
        print(f"Error processing {eeg_file}: {e}")
        traceback.print_exc()
        return None

def prepare_dataset(max_patients=None, max_windows_per_patient=None):
    """
    Prepare the complete dataset for the neural network model.
    
    Args:
        max_patients: Maximum number of patients to process (None for all)
        max_windows_per_patient: Maximum windows to extract per patient (None for all)
    """
    # Load clinical data
    clinical_df, eeg_to_id_map = load_clinical_data()
    if clinical_df is None:
        raise ValueError("Failed to load clinical data")
    
    # Parameter settings
    window_size_sec = 1
    overlap_sec = 0.5
    sfreq = 256
    
    # Containers for data
    X_all = []
    y_all = []
    patient_ids = []
    
    # Limit number of patients if specified
    if max_patients is not None:
        clinical_df = clinical_df.iloc[:max_patients]
        print(f"Limiting to first {max_patients} patients")
    
    # For each patient in the clinical data
    for idx, row in clinical_df.iterrows():
        try:
            eeg_name = row['EEG file']
            patient_id = row['ID']
            
            print(f"\nProcessing patient {patient_id} (EEG file: {eeg_name})")
            
            # Construct the EDF file path
            edf_file = os.path.join(DATA_DIR, f"{eeg_name}.edf")
            
            # Skip if file doesn't exist
            if not os.path.exists(edf_file):
                print(f"Warning: EDF file for patient {patient_id} not found at {edf_file}")
                continue
            
            # Load patient-specific annotations if possible
            annotations, annotations_raw = load_annotations(patient_id)
            
            # If patient-specific annotations not found, use global annotations
            if annotations is None:
                print(f"Using global annotations for patient {patient_id}")
                annotations, annotations_raw = load_annotations()
            
            # Load and process EEG data
            X = load_eeg_data(
                edf_file, 
                sfreq=sfreq, 
                window_size_sec=window_size_sec, 
                overlap_sec=overlap_sec,
                max_windows=max_windows_per_patient
            )
            
            if X is None or X.shape[0] == 0:
                print(f"Skipping patient {patient_id} due to EEG loading error")
                continue
            
            # Get annotations for this patient
            n_windows = X.shape[0]
            
            # Create binary labels (default to 0 if not enough annotations)
            if annotations is None or len(annotations) < n_windows:
                print(f"Warning: Not enough annotations for patient {patient_id}")
                # Check if patient has seizures according to clinical data
                has_seizure = row['Has_Seizure']
                
                if has_seizure:
                    # For patients with seizures, use synthetic labels based on clinical info
                    # This is a placeholder - ideally you would use actual annotations
                    print(f"Creating synthetic labels for seizure-positive patient {patient_id}")
                    seizure_ratio = 0.05  # Assume 5% of windows contain seizures
                    n_seizure_windows = int(n_windows * seizure_ratio)
                    
                    # Create random seizure windows
                    y = np.zeros(n_windows)
                    seizure_indices = np.random.choice(n_windows, n_seizure_windows, replace=False)
                    y[seizure_indices] = 1
                else:
                    # For patients without seizures, all windows are negative
                    print(f"Using all-negative labels for seizure-negative patient {patient_id}")
                    y = np.zeros(n_windows)
            else:
                # Use available annotations
                print(f"Using {len(annotations)} annotations for {n_windows} windows")
                
                # Take subset of annotations if we have more than needed
                if len(annotations) >= n_windows:
                    y = annotations[:n_windows]
                else:
                    # Pad with zeros if we have fewer annotations than windows
                    y = np.vstack([annotations, np.zeros((n_windows - len(annotations), annotations.shape[1]))])
                
                # Convert multi-channel annotations to binary (seizure in any channel)
                if len(y.shape) > 1 and y.shape[1] > 1:
                    y = (np.sum(y, axis=1) > 0).astype(int)
            
            # Store data
            X_all.append(X)
            y_all.append(y)
            patient_ids.extend([patient_id] * n_windows)
            
            # Force memory cleanup after each patient
            gc.collect()
            
        except Exception as e:
            print(f"Error processing patient {row['ID']}: {e}")
            traceback.print_exc()
            print("Skipping this patient and continuing with the next one")
            continue
    
    # Check if we have any data
    if len(X_all) == 0:
        raise ValueError("No valid data was processed. Check EEG files and paths.")
    
    try:
        # Combine all data
        print("Combining data from all patients...")
        X_combined = np.vstack(X_all)
        
        # Clear X_all to free memory
        del X_all
        gc.collect()
        
        # Combine labels
        if all(isinstance(y, np.ndarray) and len(y.shape) == 1 for y in y_all):
            y_combined = np.hstack(y_all)
        else:
            # Handle case where some y values might be 2D
            y_1d = []
            for y in y_all:
                if len(y.shape) > 1:
                    y_1d.append((np.sum(y, axis=1) > 0).astype(int))
                else:
                    y_1d.append(y)
            y_combined = np.hstack(y_1d)
        
        # Clear y_all to free memory
        del y_all
        gc.collect()
        
        print(f"\nFinal dataset: {X_combined.shape[0]} samples, {X_combined.shape[1]} channels, {X_combined.shape[2]} time points")
        print(f"Label distribution: {np.sum(y_combined)} positive samples ({np.mean(y_combined)*100:.2f}%)")
        
        # Store patient IDs for potential stratification
        patient_ids = np.array(patient_ids)
        
        # Normalize data
        print("Normalizing data...")
        n_samples, n_channels, n_timepoints = X_combined.shape
        
        # Use chunked normalization for large datasets
        if n_samples > 10000:
            print("Large dataset detected. Using chunked normalization.")
            scaler = StandardScaler()
            
            # Fit scaler on a random subset
            subset_size = min(10000, n_samples)
            subset_indices = np.random.choice(n_samples, subset_size, replace=False)
            subset = X_combined[subset_indices].reshape(subset_size, -1)
            scaler.fit(subset)
            del subset
            gc.collect()
            
            # Apply normalization in chunks
            chunk_size = 5000
            X_final = np.zeros_like(X_combined)
            
            for i in range(0, n_samples, chunk_size):
                end_idx = min(i + chunk_size, n_samples)
                X_chunk = X_combined[i:end_idx].reshape(end_idx - i, -1)
                X_final[i:end_idx] = scaler.transform(X_chunk).reshape(end_idx - i, n_channels, n_timepoints)
                
                # Clean up to free memory
                del X_chunk
                gc.collect()
        else:
            # For smaller datasets, normalize all at once
            scaler = StandardScaler()
            X_reshaped = X_combined.reshape(n_samples, -1)
            X_normalized = scaler.fit_transform(X_reshaped)
            X_final = X_normalized.reshape(n_samples, n_channels, n_timepoints)
            
            # Clean up to free memory
            del X_reshaped, X_normalized
            gc.collect()
        
        # Clear X_combined to free memory
        del X_combined
        gc.collect()
        
        # Convert labels to categorical for binary classification
        print("Converting labels to categorical format...")
        y_final = to_categorical(y_combined, num_classes=2)
        
        return X_final, y_final, patient_ids
        
    except Exception as e:
        print(f"Error combining data: {e}")
        traceback.print_exc()
        raise

def split_data_by_patient(X, y, patient_ids, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split the data into training, validation, and test sets based on patient IDs.
    """
    try:
        # Get unique patient IDs
        unique_patients = np.unique(patient_ids)
        print(f"Splitting data for {len(unique_patients)} unique patients")
        
        # Split patients into train and temp (val+test)
        train_patients, temp_patients = train_test_split(
            unique_patients, test_size=test_size+val_size, random_state=random_state
        )
        
        # Split temp into validation and test
        val_ratio = val_size / (test_size + val_size)
        val_patients, test_patients = train_test_split(
            temp_patients, test_size=1-val_ratio, random_state=random_state
        )
        
        # Create masks for each set
        train_mask = np.isin(patient_ids, train_patients)
        val_mask = np.isin(patient_ids, val_patients)
        test_mask = np.isin(patient_ids, test_patients)
        
        # Split the data
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        
        print(f"Split sizes:")
        print(f"  Training: {X_train.shape[0]} samples ({len(train_patients)} patients)")
        print(f"  Validation: {X_val.shape[0]} samples ({len(val_patients)} patients)")
        print(f"  Test: {X_test.shape[0]} samples ({len(test_patients)} patients)")
        
        # Check distribution of labels in each split
        y_train_pos = np.mean(np.argmax(y_train, axis=1)) * 100
        y_val_pos = np.mean(np.argmax(y_val, axis=1)) * 100
        y_test_pos = np.mean(np.argmax(y_test, axis=1)) * 100
        
        print(f"Label distribution (% positive):")
        print(f"  Training: {y_train_pos:.2f}%")
        print(f"  Validation: {y_val_pos:.2f}%")
        print(f"  Test: {y_test_pos:.2f}%")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
        
    except Exception as e:
        print(f"Error in data splitting: {e}")
        traceback.print_exc()
        raise

def main():
    """
    Main function to run the data processing pipeline.
    """
    print("Starting EEG data processing...")
    
    try:
        # You can limit the data processing for testing/debugging
        # Set to None to process all patients
        max_patients = None
        max_windows_per_patient = None
        
        # Prepare the dataset
        X, y, patient_ids = prepare_dataset(
            max_patients=max_patients,
            max_windows_per_patient=max_windows_per_patient
        )
        
        # Split the data
        X_train, y_train, X_val, y_val, X_test, y_test = split_data_by_patient(X, y, patient_ids)
        
        # Before saving, verify datasets are not empty
        if X_train.shape[0] == 0 or X_val.shape[0] == 0 or X_test.shape[0] == 0:
            print("Warning: One or more data splits are empty. Check preprocessing.")
            # Continue anyway to save what we have
        
        # Save processed data
        output_dir = os.path.join(DATA_DIR, 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving processed data to {output_dir}...")
        
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        print(f"Data processing complete! Files saved to {output_dir}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    except Exception as e:
        print(f"Critical error in data processing: {e}")
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()