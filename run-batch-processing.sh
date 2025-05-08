#!/bin/bash

# Define batch size (number of patients per batch)
BATCH_SIZE=5

# Get total number of patients from clinical data
# This is a simple approximation - modify as needed
TOTAL_PATIENTS=$(wc -l < project/data/clinical_information.csv)
TOTAL_PATIENTS=$((TOTAL_PATIENTS - 1))  # Remove header row

echo "Processing approximately $TOTAL_PATIENTS patients in batches of $BATCH_SIZE"

# Process in batches
for ((i=1; i<=$TOTAL_PATIENTS; i+=$BATCH_SIZE)); do
    end=$((i + BATCH_SIZE - 1))
    if [ $end -gt $TOTAL_PATIENTS ]; then
        end=$TOTAL_PATIENTS
    fi
    
    echo "========================================"
    echo "Processing batch: patients $i to $end"
    echo "========================================"
    
    # Run the batch processor with patient ID range
    python project/scripts/batch_eeg_processor.py $i $end
    
    # Check if the process was successful
    if [ $? -ne 0 ]; then
        echo "Error processing batch $i to $end. Stopping."
        exit 1
    fi
    
    echo "Batch $i to $end completed successfully."
    
    # Optional: add a short pause between batches
    sleep 2
done

echo "All batches processed. Combining data..."

# Run the final combination step
python project/scripts/batch_eeg_processor.py

echo "Processing complete!"