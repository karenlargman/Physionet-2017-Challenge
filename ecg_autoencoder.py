# ecg_autoencoder.py

import numpy as np
import wfdb  # For reading PhysioNet data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Load ECG Data (using wfdb)
def load_ecg_data(record_name):
    """
    Load ECG data using WFDB package from PhysioNet.
    """
    record = wfdb.rdrecord(record_name)
    return record.p_signal  # Return the ECG signal

# 2. Preprocess ECG Data (normalize and segment into windows)
def preprocess_ecg_data(ecg_data, window_size=3000):
    """
    Preprocess ECG data by normalizing and segmenting into windows.
    """
    # Normalize ECG signal using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    ecg_data = scaler.fit_transform(ecg_data.reshape(-1, 1))

    # Segment the data into windows of size `window_size`
    segments = []
    for i in range(0, len(ecg_data) - window_size, window_size):
        segment = ecg_data[i:i + window_size]
        segments.append(segment)
    return np.array(segments)

# 3. Build the Autoencoder Model (using TensorFlow/Keras)
def build_autoencoder(input_shape, latent_dim=50):
    """
    Build an autoencoder model with a specified latent dimension (e.g., 50 features).
    """
    input_layer = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Dense(128, activation='relu')(input_layer)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    latent_space = tf.keras.layers.Dense(latent_dim, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(latent_space)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    output_layer = tf.keras.layers.Dense(input_shape[0], activation='sigmoid')(x)
    
    autoencoder = tf.keras.models.Model(input_layer, output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder

# 4. Main Execution
if __name__ == '__main__':
    # Example usage
    # Load and preprocess the ECG data
    # Make sure to replace 'path_to_physionet_data' with the actual path to the data
    ecg_signal = load_ecg_data('path_to_physionet_data/A00001')  # Modify with actual path
    segments = preprocess_ecg_data(ecg_signal)

    # Split data into training and test sets (60% train, 40% test)
    X_train, X_test = train_test_split(segments, test_size=0.4, random_state=42)

    # Reshape data for autoencoder input (adding an extra dimension for channels)
    X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

    # Build and train the autoencoder
    autoencoder = build_autoencoder(input_shape=(X_train.shape[1], 1), latent_dim=50)
    autoencoder.fit(X_train_reshaped, X_train_reshaped, epochs=50, batch_size=32, validation_split=0.1)

    # Calculate the reconstruction error
    reconstructed_train_data = autoencoder.predict(X_train_reshaped)
    reconstruction_error = np.mean(np.square(X_train_reshaped - reconstructed_train_data), axis=1)
    
    # Output average reconstruction error and plot the distribution
    print(f'Average Reconstruction Error: {np.mean(reconstruction_error)}')
    
    # Plot the distribution of reconstruction errors
    plt.hist(reconstruction_error, bins=50, alpha=0.7)
    plt.title("Reconstruction Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.show()

