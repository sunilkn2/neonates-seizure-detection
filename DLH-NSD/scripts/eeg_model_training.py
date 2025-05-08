import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import tensorflow as tf

# Handle TensorFlow/Keras version differences
try:
    from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from keras.optimizers import Adam
except ImportError:
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam

# Import the model
try:
    from model import model_DL2, copy_model
    print("Successfully imported model from model.py")
except ImportError as e:
    print(f"Error importing from model.py: {e}")
    # Fallback to tensorflow.keras if needed
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from model import model_DL2, copy_model
        print("Successfully imported model after adding script directory to path")
    except ImportError as e2:
        print(f"Second import attempt failed: {e2}")
        raise

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define paths
DATA_DIR = 'project/data/processed'
MODEL_DIR = 'project/models'
RESULTS_DIR = 'project/results'

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_processed_data():
    """
    Load the preprocessed data.
    """
    try:
        X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
        X_test = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
        
        # Check if any of the files are empty
        if X_train.size == 0 or y_train.size == 0:
            raise ValueError("Training data is empty")
        
        print(f"Data loaded:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        print(f"  X_val: {X_val.shape}")
        print(f"  y_val: {y_val.shape}")
        print(f"  X_test: {X_test.shape}")
        print(f"  y_test: {y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def create_callbacks(model_name):
    """
    Create callbacks for model training.
    """
    checkpoint_path = os.path.join(MODEL_DIR, f"{model_name}_best.h5")
    
    callbacks = [
        ModelCheckpoint(
            checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode='min',
            verbose=1
        )
    ]
    
    return callbacks

def train_model(X_train, y_train, X_val, y_val, model_name='eeg_inception1d'):
    """
    Train the model using the processed data.
    """
    # Set hyperparameters
    weight_decay = 0.005
    learning_rate = 0.001
    lr_decay = 1e-4
    batch_size = 32
    epochs = 50  # Reduced from 100 to speed up training
    
    # Print input shapes to debug
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    try:
        # Initialize the model
        model = model_DL2(
            trainX=X_train,
            trainy=y_train,
            wd=weight_decay,
            lr=learning_rate,
            lr_decay=lr_decay
        )
        
        # Compile the model - choose loss based on output shape
        try:
            optimizer = Adam(learning_rate=learning_rate, decay=lr_decay)
        except:
            # For older Keras versions
            optimizer = Adam(lr=learning_rate, decay=lr_decay)
        
        # Determine appropriate loss function based on output shape
        if y_train.shape[1] > 1:  # Multi-class (one-hot encoded)
            loss = 'categorical_crossentropy'
            print("Using categorical_crossentropy loss for multi-class output")
        else:  # Binary
            loss = 'binary_crossentropy'
            print("Using binary_crossentropy loss for binary output")
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        # Create callbacks
        callbacks = create_callbacks(model_name)
        
        # Train the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        model_path = os.path.join(MODEL_DIR, f"{model_name}_final.h5")
        model.save(model_path)
        
        print(f"Model saved to {model_path}")
        
        return model, history
    
    except Exception as e:
        print(f"Error in model training: {e}")
        import traceback
        traceback.print_exc()
        raise

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on the test set and generate performance metrics.
    """
    # Get predictions
    y_pred_prob = model.predict(X_test)
    
    # Handle different output shapes
    if len(y_pred_prob.shape) > 1 and y_pred_prob.shape[1] > 1:
        # Multi-class output
        y_pred_class = np.argmax(y_pred_prob, axis=1)
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test_class = np.argmax(y_test, axis=1)
        else:
            y_test_class = y_test
            
        # For ROC curve, use the probability of the positive class
        y_pred_prob_positive = y_pred_prob[:, 1]
    else:
        # Binary output
        y_pred_prob = y_pred_prob.flatten()
        y_pred_class = (y_pred_prob > 0.5).astype(int)
        
        if len(y_test.shape) > 1 and y_test.shape[1] > 1:
            y_test_class = np.argmax(y_test, axis=1)
        else:
            y_test_class = y_test
            
        y_pred_prob_positive = y_pred_prob
    
    # Classification report
    report = classification_report(y_test_class, y_pred_class, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test_class, y_pred_class)
    
    # ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_test_class, y_pred_prob_positive)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test_class, y_pred_prob_positive)
    
    # Save results
    results = {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc},
        'pr': {'precision': precision.tolist(), 'recall': recall.tolist()}
    }
    
    return results

def plot_results(history, results, model_name='eeg_inception1d'):
    """
    Plot training history and evaluation results.
    """
    try:
        # Check what metrics are available in history
        print("Available metrics in history:", history.history.keys())
        
        # Plot training history with error handling
        plt.figure(figsize=(12, 4))
        
        # Left plot - Loss
        plt.subplot(1, 2, 1)
        plt.plot(history.history.get('loss', []), label='Training Loss')
        plt.plot(history.history.get('val_loss', []), label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Right plot - Accuracy (with fallback options)
        plt.subplot(1, 2, 2)
        if 'accuracy' in history.history:
            plt.plot(history.history['accuracy'], label='Training Accuracy')
            if 'val_accuracy' in history.history:
                plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        elif 'acc' in history.history:
            plt.plot(history.history['acc'], label='Training Accuracy')
            if 'val_acc' in history.history:
                plt.plot(history.history['val_acc'], label='Validation Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_training_history.png"))
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = np.array(results['confusion_matrix'])
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        classes = ['No Seizure', 'Seizure']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png"))
    except Exception as e:
        print(f"Error in plotting: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(results['roc']['fpr'], results['roc']['tpr'], 
                 label=f'ROC curve (area = {results["roc"]["auc"]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_roc_curve.png"))
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(8, 6))
        plt.plot(results['pr']['recall'], results['pr']['precision'])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_pr_curve.png"))
    except Exception as e:
        print(f"Error in ROC/PR curve plotting: {e}")
    
def main():
    """
    Main function to run the training pipeline.
    """
    try:
        print("Loading processed data...")
        X_train, y_train, X_val, y_val, X_test, y_test = load_processed_data()
        
        print("Training model...")
        model_name = 'eeg_inception1d'
        model, history = train_model(X_train, y_train, X_val, y_val, model_name)
        
        print("Evaluating model...")
        results = evaluate_model(model, X_test, y_test)
        
        print("Plotting results...")
        plot_results(history, results, model_name)
        
        # Save results to JSON
        import json
        with open(os.path.join(RESULTS_DIR, f"{model_name}_results.json"), 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"Results saved to {RESULTS_DIR}")
        
        # Print classification report
        print("\nClassification Report:")
        report = results['classification_report']
        
        for cls in report:
            if cls in ['0', '1']:  # Skip macro/weighted avg
                class_name = 'No Seizure' if cls == '0' else 'Seizure'
                print(f"{class_name}:")
                for metric, value in report[cls].items():
                    print(f"  {metric}: {value:.4f}")
        
        print(f"\nAUC: {results['roc']['auc']:.4f}")
        
    except Exception as e:
        print(f"Error in main: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()