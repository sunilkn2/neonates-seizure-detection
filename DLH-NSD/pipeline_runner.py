import os
import argparse
import sys
import subprocess
import logging
import json
import time
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eeg_pipeline.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("EEG-Pipeline")

# Define paths
PROJECT_DIR = os.path.abspath('project')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODEL_DIR = os.path.join(PROJECT_DIR, 'models')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
VIZ_DIR = os.path.join(PROJECT_DIR, 'visualizations')

# Ensure directories exist
for directory in [DATA_DIR, PROCESSED_DIR, MODEL_DIR, RESULTS_DIR, VIZ_DIR]:
    os.makedirs(directory, exist_ok=True)

def run_module(module_path, description):
    """
    Run a Python module and handle errors.
    """
    logger.info(f"Starting: {description}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            [sys.executable, module_path],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        logger.info(f"Output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Stderr: {result.stderr}")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Completed: {description} in {elapsed_time:.2f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in {description}: {e}")
        logger.error(f"Stdout: {e.stdout}")
        logger.error(f"Stderr: {e.stderr}")
        return False

def save_run_config(args):
    """
    Save the run configuration to a JSON file.
    """
    config = vars(args)
    config['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    config_path = os.path.join(RESULTS_DIR, f"run_config_{int(time.time())}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    logger.info(f"Run configuration saved to {config_path}")
    return config_path

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="EEG Analysis Pipeline Runner")
    
    parser.add_argument("--skip-preprocessing", action="store_true", 
                        help="Skip data preprocessing step")
    parser.add_argument("--skip-training", action="store_true", 
                        help="Skip model training step")
    parser.add_argument("--skip-visualization", action="store_true", 
                        help="Skip visualization step")
    
    parser.add_argument("--model-name", default="eeg_inception1d",
                        help="Name of the model to use/train")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate for training")
    parser.add_argument("--weight-decay", type=float, default=0.005,
                        help="Weight decay (L2 regularization)")
    
    return parser.parse_args()

def main():
    """
    Main function to run the complete pipeline.
    """
    args = parse_arguments()
    config_path = save_run_config(args)
    
    logger.info("Starting EEG Analysis Pipeline")
    logger.info(f"Project directory: {PROJECT_DIR}")
    
    # Path to Python modules
    data_processor_path = os.path.join(PROJECT_DIR, "scripts", "eeg_data_processor.py")
    model_trainer_path = os.path.join(PROJECT_DIR, "scripts", "eeg_model_training.py")
    visualizer_path = os.path.join(PROJECT_DIR, "scripts", "eeg_visualization.py")
    
    # Create a parameters file for the modules to use
    params = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "data_dir": DATA_DIR,
        "processed_dir": PROCESSED_DIR,
        "model_dir": MODEL_DIR,
        "results_dir": RESULTS_DIR,
        "viz_dir": VIZ_DIR
    }
    
    params_path = os.path.join(PROJECT_DIR, "params.json")
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
    
    logger.info(f"Parameters saved to {params_path}")
    
    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        success = run_module(data_processor_path, "Data Preprocessing")
        if not success:
            logger.error("Data preprocessing failed. Pipeline stopped.")
            return False
    else:
        logger.info("Skipping data preprocessing step")
    
    # Step 2: Model Training
    if not args.skip_training:
        success = run_module(model_trainer_path, "Model Training")
        if not success:
            logger.error("Model training failed. Pipeline stopped.")
            return False
    else:
        logger.info("Skipping model training step")
    
    # Step 3: Visualization and Analysis
    if not args.skip_visualization:
        success = run_module(visualizer_path, "Visualization and Analysis")
        if not success:
            logger.warning("Visualization step failed but the pipeline will continue.")
    else:
        logger.info("Skipping visualization step")
    
    logger.info("EEG Analysis Pipeline completed successfully!")
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Unhandled exception in pipeline: {e}")
        sys.exit(1)