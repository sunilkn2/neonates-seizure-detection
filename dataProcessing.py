import mne
import numpy as np
import pandas as pd
import os
from scipy.io import loadmat # If annotations are in .mat format

# --- Configuration ---
DATASET_ROOT = '/path/to/your/helsinki_dataset_root/' # IMPORTANT: Change this path
OUTPUT_DIR = '/path/to/your/processed_data_output/' # IMPORTANT: Change this path
PATIENT_IDS = ['pat_01', 'pat_02', ...] # TODO: Populate with the 39 consensus patient IDs or all IDs you want to process

TARGET_SFREQ = 32
EPOCH_DURATION_S = 8
EPOCH_OVERLAP_S = 4
CONSENSUS_THRESHOLD = 0.8 # Proportion of seizure seconds in epoch to label as seizure

# Define the 18 bipolar channels for the Helsinki dataset montage
HELSINKI_BIPOLAR_PAIRS = [
    ('Fp2', 'F4'), ('F4', 'C4'), ('C4', 'P4'), ('P4', 'O2'),
    ('Fp1', 'F3'), ('F3', 'C3'), ('C3', 'P3'), ('P3', 'O1'),
    ('Fp2', 'F8'), ('F8', 'T4'), ('T4', 'T6'), ('T6', 'O2'),
    ('Fp1', 'F7'), ('F7', 'T3'), ('T3', 'T5'), ('T5', 'O1'),
    ('Fz', 'Cz'), ('Cz', 'Pz')
]
HELSINKI_BIPOLAR_CH_NAMES = [f"{p[0]}-{p[1]}" for p in HELSINKI_BIPOLAR_PAIRS]
HELSINKI_ANODES = [p[0] for p in HELSINKI_BIPOLAR_PAIRS]
HELSINKI_CATHODES = [p[1] for p in HELSINKI_BIPOLAR_PAIRS]

# --- Helper Functions ---

def load_annotations_csv(filepath):
    """Loads annotations from CSV, returns dict {rater_id: [(start_s, end_s), ...]}. """
    try:
        df = pd.read_csv(filepath)
        # Assuming columns 'Rater', 'Start_s', 'End_s' - adjust if needed
        annotations = {}
        for rater, group in df.groupby('Rater'):
            annotations[f'Rater_{rater}'] = list(zip(group['Start_s'], group['End_s']))
        # Simple check for 3 raters often assumed in paper
        if len(annotations) < 3:
             print(f"Warning: Found {len(annotations)} raters in {filepath}, expected 3 for consensus.")
        return annotations
    except FileNotFoundError:
        print(f"Annotation file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading annotation CSV {filepath}: {e}")
        return None

def load_annotations_mat(filepath):
    """Loads annotations from MAT, returns dict {rater_id: [(start_s, end_s), ...]}. """
    try:
        mat_data = loadmat(filepath)
        # --- !!! Requires Inspection of the .mat file structure !!! ---
        # This is a placeholder structure - you MUST adapt it based on the actual .mat file
        annotations = {}
        # Example: Iterate through potential rater fields if structured that way
        for key in mat_data:
             if 'rater' in key.lower() and isinstance(mat_data[key], np.ndarray):
                  # Assuming Nx2 array [start_s, end_s]
                  events = mat_data[key].tolist()
                  annotations[key] = [(row[0], row[1]) for row in events] # Ensure correct indexing
        if not annotations:
             print(f"Warning: Could not parse rater events from MAT file: {filepath}")
             return None
        if len(annotations) < 3:
             print(f"Warning: Found {len(annotations)} raters in {filepath}, expected 3.")
        return annotations

    except FileNotFoundError:
        print(f"Annotation file not found: {filepath}")
        return None
    except Exception as e:
        print(f"Error loading annotation MAT {filepath}: {e}")
        return None


def create_consensus_mask(rater_annotations, n_samples, sfreq):
    """Creates a boolean mask where True indicates all raters marked seizure."""
    if not rater_annotations or len(rater_annotations) < 3:
         print("Insufficient rater annotations for consensus.")
         # Return all False mask, or handle as error depending on use case
         return np.zeros(n_samples, dtype=bool)

    rater_masks = []
    num_raters = len(rater_annotations)
    for rater_id, events in rater_annotations.items():
        mask = np.zeros(n_samples, dtype=bool)
        for start_s, end_s in events:
            # Ensure indices are within bounds
            start_sample = max(0, int(np.round(start_s * sfreq)))
            end_sample = min(n_samples, int(np.round(end_s * sfreq)))
            if start_sample < end_sample: # Avoid empty slices
                mask[start_sample:end_sample] = True
        rater_masks.append(mask)

    # Consensus: True only if ALL raters agree
    consensus_mask = np.all(rater_masks, axis=0)
    return consensus_mask


def process_patient(patient_id, dataset_root, output_dir):
    """Processes a single patient's EEG data and annotations."""
    print(f"Processing patient: {patient_id}...")
    edf_path = os.path.join(dataset_root, patient_id, f"{patient_id}.edf")
    # Try loading CSV first, then MAT for annotations
    annot_path_csv = os.path.join(dataset_root, patient_id, f"{patient_id}_annotations.csv")
    annot_path_mat = os.path.join(dataset_root, patient_id, f"{patient_id}_annotations.mat") # Adjust filename if different

    if os.path.exists(annot_path_csv):
        rater_annotations = load_annotations_csv(annot_path_csv)
    elif os.path.exists(annot_path_mat):
        rater_annotations = load_annotations_mat(annot_path_mat)
    else:
        print(f"No annotation file found for patient {patient_id}")
        return None

    if not rater_annotations:
        print(f"Could not load annotations for {patient_id}")
        return None

    # 1. Load Raw EDF
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except FileNotFoundError:
        print(f"EDF file not found: {edf_path}")
        return None
    except Exception as e:
         print(f"Error loading EDF {edf_path}: {e}")
         return None

    # Check if required channels for montage exist
    required_raw_channels = set(HELSINKI_ANODES + HELSINKI_CATHODES)
    missing_channels = required_raw_channels - set(raw.ch_names)
    if missing_channels:
        print(f"Warning: Patient {patient_id} missing channels needed for montage: {missing_channels}")
        # Option: Skip patient or try to proceed if only minor channels missing? For now, skip.
        # return None

    # 2. Apply Bipolar Montage
    try:
        raw_bi = mne.set_bipolar_reference(raw,
                                           anode=HELSINKI_ANODES,
                                           cathode=HELSINKI_CATHODES,
                                           ch_name=HELSINKI_BIPOLAR_CH_NAMES,
                                           copy=True,
                                           verbose=False)
        # Drop original channels, keep only bipolar
        raw_bi.pick_channels(HELSINKI_BIPOLAR_CH_NAMES)
    except Exception as e:
        print(f"Error applying montage to {patient_id}: {e}")
        return None


    # 3. Filter
    try:
        raw_bi.notch_filter(freqs=50, verbose=False) # 50 Hz for Europe
        raw_bi.filter(l_freq=0.5, h_freq=16.0, verbose=False)
    except Exception as e:
        print(f"Error filtering {patient_id}: {e}")
        return None

    # 4. Resample
    try:
        raw_bi.resample(sfreq=TARGET_SFREQ, verbose=False)
        current_sfreq = raw_bi.info['sfreq']
    except Exception as e:
        print(f"Error resampling {patient_id}: {e}")
        return None


    # 5. Get Data Array
    eeg_data, times = raw_bi.get_data(return_times=True) # Shape (18, N_samples)
    n_chans, n_samples = eeg_data.shape

    # 6. Create Consensus Annotation Mask (per sample)
    consensus_mask = create_consensus_mask(rater_annotations, n_samples, current_sfreq)

    # 7. Epoch Data and Annotations
    epoch_len_samples = int(EPOCH_DURATION_S * current_sfreq) # 8 * 32 = 256
    step_samples = int((EPOCH_DURATION_S - EPOCH_OVERLAP_S) * current_sfreq) # 4 * 32 = 128

    data_epochs = []
    label_epochs = []
    start = 0
    while start + epoch_len_samples <= n_samples:
        # Data epoch
        data_epoch = eeg_data[:, start : start + epoch_len_samples]
        data_epochs.append(data_epoch)

        # Label epoch
        epoch_mask_segment = consensus_mask[start : start + epoch_len_samples]
        seizure_proportion = np.mean(epoch_mask_segment)
        label_epochs.append(seizure_proportion > CONSENSUS_THRESHOLD)

        start += step_samples

    if not data_epochs:
        print(f"Patient {patient_id}: No full epochs generated.")
        return None

    data_epochs_np = np.stack(data_epochs) # Shape (N_epochs, N_chans, N_samples_per_epoch)
    label_epochs_np = np.array(label_epochs, dtype=int) # Shape (N_epochs,)

    # 8. Clean NaN Epochs (Should be minimal after MNE processing)
    nan_mask = np.isnan(data_epochs_np).any(axis=(1, 2))
    if np.any(nan_mask):
        print(f"Patient {patient_id}: Found {np.sum(nan_mask)} NaN epochs, removing.")
        data_epochs_clean = data_epochs_np[~nan_mask]
        label_epochs_clean = label_epochs_np[~nan_mask]
    else:
        data_epochs_clean = data_epochs_np
        label_epochs_clean = label_epochs_np

    print(f"Patient {patient_id}: Generated {data_epochs_clean.shape[0]} epochs. "
          f"({np.sum(label_epochs_clean)} seizure epochs)")

    # 9. Save Processed Data
    output_patient_dir = os.path.join(output_dir, patient_id)
    os.makedirs(output_patient_dir, exist_ok=True)
    np.save(os.path.join(output_patient_dir, f"{patient_id}_data_epochs.npy"), data_epochs_clean)
    np.save(os.path.join(output_patient_dir, f"{patient_id}_label_epochs.npy"), label_epochs_clean)

    print(f"Successfully processed and saved data for {patient_id}")
    return patient_id # Indicate success

# --- Main Execution ---
if __name__ == "__main__":
    if not os.path.isdir(DATASET_ROOT):
        print(f"Error: Dataset root directory not found: {DATASET_ROOT}")
        exit()
    if not PATIENT_IDS:
         print("Error: PATIENT_IDS list is empty. Please populate it.")
         # Example: Populate PATIENT_IDS by listing directories in DATASET_ROOT
         # PATIENT_IDS = [d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))]
         # print(f"Auto-detected {len(PATIENT_IDS)} potential patient folders.")
         # exit() # Remove exit if you want to proceed with auto-detected IDs

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    processed_patients = []
    failed_patients = []

    for pid in PATIENT_IDS:
        result = process_patient(pid, DATASET_ROOT, OUTPUT_DIR)
        if result:
            processed_patients.append(result)
        else:
            failed_patients.append(pid)

    print("\n--- Processing Summary ---")
    print(f"Successfully processed: {len(processed_patients)} patients.")
    if failed_patients:
        print(f"Failed to process: {len(failed_patients)} patients: {failed_patients}")
    print(f"Processed data saved to: {OUTPUT_DIR}")
