"""
Deep learning model for fraud detection in banking transactions.
Includes both classification and autoencoder-based anomaly detection approaches.
"""
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import os

# Check for GPU availability
def check_gpu():
    """Check if GPU is available and print device information."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU(s) detected: {len(gpus)}")
        for gpu in gpus:
            print(f" - {gpu}")
        return True
    else:
        print("No GPU detected. Using CPU for model training.")
        return False

def create_classification_model(input_dim, hidden_layers=[64, 32], dropout_rate=0.4):
    """
    Create a feedforward neural network for binary fraud classification.
    
    Args:
        input_dim (int): Number of input features
        hidden_layers (list): List of hidden layer sizes
        dropout_rate (float): Dropout rate for regularization
        
    Returns:
        Model: Compiled Keras model
    """
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    # Add hidden layers
    for units in hidden_layers:
        model.add(layers.Dense(units, activation='relu'))
        model.add(layers.Dropout(dropout_rate))
    
    # Output layer (binary classification)
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def create_autoencoder_model(input_dim, encoding_dim=16, hidden_layers=[64, 32]):
    """
    Create an autoencoder model for unsupervised anomaly detection.
    
    Args:
        input_dim (int): Number of input features
        encoding_dim (int): Size of the encoded representation
        hidden_layers (list): List of hidden layer sizes for encoder (reversed for decoder)
        
    Returns:
        Model: Compiled Keras autoencoder model
    """
    # Encoder
    encoder_inputs = tf.keras.Input(shape=(input_dim,))
    x = encoder_inputs
    
    # Add encoder hidden layers
    for units in hidden_layers:
        x = layers.Dense(units, activation='relu')(x)
    
    # Encoded representation
    encoded = layers.Dense(encoding_dim, activation='relu', name='encoded')(x)
    
    # Decoder
    x = encoded
    
    # Add decoder hidden layers (reverse of encoder)
    for units in reversed(hidden_layers):
        x = layers.Dense(units, activation='relu')(x)
    
    # Output layer (reconstruction)
    decoded = layers.Dense(input_dim, activation='linear')(x)
    
    # Create and compile model
    autoencoder = models.Model(encoder_inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

def train_classification_model(X_train, y_train, X_val=None, y_val=None, 
                               input_dim=None, batch_size=256, epochs=10, 
                               class_weight=None, model_path=None):
    """
    Train a classification model for fraud detection.
    
    Args:
        X_train (array): Training features
        y_train (array): Training labels
        X_val (array, optional): Validation features
        y_val (array, optional): Validation labels
        input_dim (int, optional): Input dimension (inferred from X_train if None)
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        class_weight (dict, optional): Class weights for imbalanced data
        model_path (str, optional): Path to save the trained model
        
    Returns:
        Model: Trained Keras model
        History: Training history
    """
    if input_dim is None:
        input_dim = X_train.shape[1]
    
    # Create model
    model = create_classification_model(input_dim)
    
    # Define callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    ]
    
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        callbacks_list.append(
            callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
        )
    
    # Train model
    validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
    
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=callbacks_list,
        class_weight=class_weight
    )
    
    return model, history

def train_autoencoder_model(X_train, X_val=None, input_dim=None, 
                            batch_size=256, epochs=20, model_path=None):
    """
    Train an autoencoder model for anomaly detection.
    
    Args:
        X_train (array): Training features (normal transactions only)
        X_val (array, optional): Validation features
        input_dim (int, optional): Input dimension (inferred from X_train if None)
        batch_size (int): Batch size for training
        epochs (int): Number of training epochs
        model_path (str, optional): Path to save the trained model
        
    Returns:
        Model: Trained Keras autoencoder model
        History: Training history
    """
    if input_dim is None:
        input_dim = X_train.shape[1]
    
    # Create model
    model = create_autoencoder_model(input_dim)
    
    # Define callbacks
    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', patience=5),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)
    ]
    
    if model_path:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        callbacks_list.append(
            callbacks.ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
        )
    
    # Train model
    validation_data = X_val if X_val is not None else None
    
    history = model.fit(
        X_train, X_train,  # Autoencoder tries to reconstruct the input
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(validation_data, validation_data) if validation_data is not None else None,
        callbacks=callbacks_list
    )
    
    return model, history

def compute_anomaly_scores(model, X):
    """
    Compute anomaly scores using the autoencoder reconstruction error.
    
    Args:
        model (Model): Trained autoencoder model
        X (array): Input data
        
    Returns:
        array: Anomaly scores (reconstruction errors)
    """
    # Get reconstructions
    X_pred = model.predict(X)
    
    # Compute mean squared error for each sample
    mse = np.mean(np.square(X - X_pred), axis=1)
    
    return mse

if __name__ == "__main__":
    # Check for GPU availability
    has_gpu = check_gpu()
    
    # Placeholder for loading preprocessed data and training the model
    # X_train, y_train = ...
    # model = create_classification_model(input_dim=X_train.shape[1])
    # model.fit(X_train, y_train, epochs=10, batch_size=256, validation_split=0.2)
