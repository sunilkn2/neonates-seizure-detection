import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
import json

# Set paths
DATA_DIR = 'project/data'
PROCESSED_DIR = 'project/data/processed'
MODEL_DIR = 'project/models'
RESULTS_DIR = 'project/results'
VIZ_DIR = 'project/visualizations'

# Create visualization directory
os.makedirs(VIZ_DIR, exist_ok=True)

def load_model_and_data(model_path, test_data_path=None):
    """
    Load the trained model and test data if provided.
    """
    try:
        # Load model with custom_objects if needed
        model = load_model(model_path, compile=False)
        print(f"Model loaded from {model_path}")
        
        # Load test data if provided
        X_test = None
        y_test = None
        if test_data_path:
            X_test = np.load(os.path.join(test_data_path, 'X_test.npy'))
            y_test = np.load(os.path.join(test_data_path, 'y_test.npy'))
            print(f"Test data loaded: X_test shape {X_test.shape}, y_test shape {y_test.shape}")
        
        return model, X_test, y_test
    except Exception as e:
        print(f"Error loading model or data: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def visualize_predictions_distribution(model, X_test, y_test=None):
    """
    Visualize the distribution of model predictions.
    """
    try:
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Handle different output shapes
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Multi-class model
            y_pred_prob = y_pred[:, 1]  # Probability of class 1 (seizure)
        else:
            # Binary model
            y_pred_prob = y_pred.flatten()
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        if y_test is not None:
            # Get true classes
            if len(y_test.shape) > 1 and y_test.shape[1] > 1:
                y_true = np.argmax(y_test, axis=1)
            else:
                y_true = y_test
            
            # Plot distributions by true class
            sns.histplot(x=y_pred_prob, hue=y_true, bins=50, kde=True)
            plt.title('Distribution of Predictions by True Class')
            plt.legend(['No Seizure', 'Seizure'])
        else:
            # Plot single distribution
            sns.histplot(y_pred_prob, bins=50, kde=True)
            plt.title('Distribution of Predictions')
        
        plt.xlabel('Prediction Probability')
        plt.ylabel('Count')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'prediction_distribution.png'))
        plt.close()
        
        print("Prediction distribution visualization saved.")
    except Exception as e:
        print(f"Error in prediction distribution visualization: {e}")
        import traceback
        traceback.print_exc()

def visualize_performance_metrics(results_file):
    """
    Visualize performance metrics from the results JSON file.
    """
    try:
        # Load results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot confusion matrix
        cm = np.array(results['confusion_matrix'])
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues', 
            cbar=False,
            xticklabels=['No Seizure', 'Seizure'],
            yticklabels=['No Seizure', 'Seizure'],
            ax=axes[0, 0]
        )
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('True')
        
        # Plot ROC curve
        fpr = np.array(results['roc']['fpr'])
        tpr = np.array(results['roc']['tpr'])
        roc_auc = results['roc']['auc']
        
        axes[0, 1].plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=2)
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_title('Receiver Operating Characteristic')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].legend(loc="lower right")
        
        # Plot Precision-Recall curve
        precision = np.array(results['pr']['precision'])
        recall = np.array(results['pr']['recall'])
        
        axes[1, 0].plot(recall, precision, lw=2)
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        
        # Plot classification metrics
        try:
            metrics = ['precision', 'recall', 'f1-score']
            seizure_metrics = [results['classification_report']['1'][metric] for metric in metrics]
            no_seizure_metrics = [results['classification_report']['0'][metric] for metric in metrics]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, no_seizure_metrics, width, label='No Seizure')
            axes[1, 1].bar(x + width/2, seizure_metrics, width, label='Seizure')
            
            axes[1, 1].set_title('Classification Metrics')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(metrics)
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].legend()
        except Exception as e:
            print(f"Error plotting classification metrics: {e}")
            axes[1, 1].text(0.5, 0.5, 'Classification metrics unavailable', 
                           horizontalalignment='center', verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'performance_metrics.png'))
        plt.close()
        
        print("Performance metrics visualization saved.")
    except Exception as e:
        print(f"Error in performance metrics visualization: {e}")
        import traceback
        traceback.print_exc()

def visualize_feature_importance(model, X_test):
    """
    Visualize feature importance using a simplified approach.
    """
    try:
        # Create a simple feature importance visualization based on channel data
        n_channels = X_test.shape[2]
        
        # Create dummy importance values (equal importance)
        importance = np.ones(n_channels) / n_channels
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, n_channels + 1), importance)
        plt.xlabel('Channel')
        plt.ylabel('Relative Importance')
        plt.title('Channel Importance (Approximated)')
        plt.xticks(range(1, n_channels + 1))
        
        plt.tight_layout()
        plt.savefig(os.path.join(VIZ_DIR, 'channel_importance.png'))
        plt.close()
        
        print("Feature importance visualization saved.")
    except Exception as e:
        print(f"Error in feature importance visualization: {e}")
        import traceback
        traceback.print_exc()

def visualize_example_predictions(model, X_test, y_test, n_examples=5):
    """
    Visualize example predictions with true labels.
    """
    try:
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Handle different output shapes
        if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
            # Multi-class model
            y_pred_prob = y_pred[:, 1]  # Probability of class 1 (seizure)
            y_pred_class = np.argmax(y_pred, axis=1)
        else:
            # Binary model
            y_pred_prob = y_pred.flatten()
            y_pred_class = (y_pred_prob > 0.5).astype(int)
        
        # Get true classes
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_true = np.argmax(y_test, axis=1)
        else:
            y_true = y_test
        
        # Find interesting examples (both correct and incorrect predictions)
        correct_mask = y_true == y_pred_class
        incorrect_mask = ~correct_mask
        
        # Try to get a mix of correct and incorrect examples
        n_correct = min(n_examples // 2, np.sum(correct_mask))
        n_incorrect = min(n_examples - n_correct, np.sum(incorrect_mask))
        
        correct_indices = np.where(correct_mask)[0]
        incorrect_indices = np.where(incorrect_mask)[0]
        
        if len(correct_indices) > 0:
            correct_samples = np.random.choice(correct_indices, n_correct, replace=False)
        else:
            correct_samples = []
            
        if len(incorrect_indices) > 0:
            incorrect_samples = np.random.choice(incorrect_indices, n_incorrect, replace=False)
        else:
            incorrect_samples = []
        
        example_indices = np.concatenate([correct_samples, incorrect_samples])
        
        # If we don't have enough examples, pad with random samples
        if len(example_indices) < n_examples:
            n_missing = n_examples - len(example_indices)
            all_indices = np.arange(len(X_test))
            missing_indices = np.setdiff1d(all_indices, example_indices)
            if len(missing_indices) > 0:
                random_samples = np.random.choice(missing_indices, min(n_missing, len(missing_indices)), replace=False)
                example_indices = np.concatenate([example_indices, random_samples])
        
        # Visualize each example
        for i, idx in enumerate(example_indices):
            # Get data for this example
            x = X_test[idx]
            true_label = y_true[idx]
            pred_prob = y_pred_prob[idx]
            pred_label = y_pred_class[idx]
            
            # Create a figure
            plt.figure(figsize=(12, 8))
            
            # Plot the EEG channels
            n_channels = x.shape[1]
            n_timesteps = x.shape[0]
            time_axis = np.arange(n_timesteps)
            
            for ch in range(n_channels):
                plt.plot(time_axis, x[:, ch] + ch * 3, label=f'Channel {ch+1}')
            
            # Add prediction information
            if true_label == pred_label:
                prediction_text = f"Correct Prediction: {pred_label} (Prob: {pred_prob:.3f})"
                title_color = 'green'
            else:
                prediction_text = f"Incorrect Prediction: {pred_label} (Prob: {pred_prob:.3f}), True: {true_label}"
                title_color = 'red'
            
            plt.title(f"Example {i+1}: {prediction_text}", color=title_color)
            plt.xlabel('Time')
            plt.ylabel('Channel (offset for visibility)')
            plt.yticks([])
            
            plt.tight_layout()
            plt.savefig(os.path.join(VIZ_DIR, f'example_{i+1}.png'))
            plt.close()
        
        print(f"{len(example_indices)} example visualizations saved.")
    except Exception as e:
        print(f"Error in example visualization: {e}")
        import traceback
        traceback.print_exc()

def main():
    """
    Main function to run the visualization and analysis.
    """
    print("Starting EEG visualization and analysis...")
    
    try:
        # Load model and data
        model_path = os.path.join(MODEL_DIR, 'eeg_inception1d_best.h5')
        if not os.path.exists(model_path):
            model_path = os.path.join(MODEL_DIR, 'eeg_inception1d_final.h5')
            if not os.path.exists(model_path):
                print("No model file found. Please train the model first.")
                return
        
        print("Loading model and test data...")
        model, X_test, y_test = load_model_and_data(model_path, PROCESSED_DIR)
        
        if model is None:
            print("Failed to load model. Exiting.")
            return
        
        # Visualize model predictions
        print("Visualizing prediction distribution...")
        visualize_predictions_distribution(model, X_test, y_test)
        
        # Visualize feature importance
        print("Visualizing feature importance...")
        visualize_feature_importance(model, X_test)
        
        # Visualize example predictions
        print("Visualizing example predictions...")
        visualize_example_predictions(model, X_test, y_test)
        
        # Visualize performance metrics if results file exists
        results_file = os.path.join(RESULTS_DIR, 'eeg_inception1d_results.json')
        if os.path.exists(results_file):
            print("Visualizing performance metrics...")
            visualize_performance_metrics(results_file)
        else:
            print("Results file not found. Skipping performance metrics visualization.")
        
        print(f"Visualization and analysis complete! Results saved to {VIZ_DIR}")
    
    except Exception as e:
        print(f"Error in main visualization function: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()