import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, Concatenate, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2

def copy_model(model_origin, model_target):
    """Copy weights from one model to another."""
    for l_orig, l_targ in zip(model_origin.layers, model_target.layers):
        l_targ.set_weights(l_orig.get_weights())
    return model_target

def simple_inception_block(x, filters=16):
    """A simplified inception block."""
    # 1x1 convolution
    conv1 = Conv1D(filters, 1, padding='same', activation='relu')(x)
    
    # 1x1 -> 3x3 convolution path
    conv3 = Conv1D(filters, 1, padding='same', activation='relu')(x)
    conv3 = Conv1D(filters, 3, padding='same', activation='relu')(conv3)
    
    # 1x1 -> 5x5 convolution path
    conv5 = Conv1D(filters, 1, padding='same', activation='relu')(x)
    conv5 = Conv1D(filters, 5, padding='same', activation='relu')(conv5)
    
    # Max pooling path
    pool = MaxPooling1D(3, strides=1, padding='same')(x)
    pool = Conv1D(filters, 1, padding='same', activation='relu')(pool)
    
    # Concatenate all paths
    return Concatenate(axis=-1)([conv1, conv3, conv5, pool])

def Inception1D(input_shape=(256, 1)):
    """Simple Inception1D model without any complex layers."""
    # This function is kept for compatibility but not used in this simplified version
    return None

def model_DL2(trainX, trainy, wd=0.005, lr=0.01, lr_decay=1e-4):
    """
    A simplified EEG classification model without Lambda layers.
    Works directly with data in format (batch, time, channels).
    
    Args:
        trainX: Training data with shape (n_samples, n_timesteps, n_channels)
        trainy: Training labels
        wd: Weight decay
        lr: Learning rate
        lr_decay: Learning rate decay
        
    Returns:
        model: Keras model
    """
    print(f"Building simplified model with input shape: {trainX.shape}")
    print(f"Label shape: {trainy.shape}")
    
    # Input dimensions
    n_timesteps, n_channels = trainX.shape[1], trainX.shape[2]
    n_outputs = trainy.shape[1]  # Number of output classes
    
    print(f"Output shape will be: {n_outputs}")
    
    # Input layer
    inputs = Input(shape=(n_timesteps, n_channels))
    
    # Initial preprocessing
    x = Conv1D(32, 3, padding='same', activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    # First inception block
    x = simple_inception_block(x, filters=16)
    x = MaxPooling1D(2)(x)
    
    # Second inception block
    x = simple_inception_block(x, filters=32)
    x = MaxPooling1D(2)(x)
    
    # Global pooling
    x = GlobalAveragePooling1D()(x)
    
    # Dense layers
    x = Dense(64, activation='relu', kernel_regularizer=l2(wd))(x)
    x = Dropout(0.5)(x)
    
    # Output layer - match the shape of trainy
    if n_outputs > 1:
        # Multi-class output (for one-hot encoded labels)
        outputs = Dense(n_outputs, activation='softmax')(x)
    else:
        # Binary output (for binary labels)
        outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model