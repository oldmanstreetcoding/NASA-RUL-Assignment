# Import necessary libraries
import time
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Function to create LSTM or GRU models
def create_model(model_type, units, dropout_rate, optimizer, sequence_array, label_array):
    """
    Build the LSTM or GRU model structure.

    Parameters:
    - model_type: str, either 'LSTM' or 'GRU' to select the type of recurrent model.
    - units: int, the number of units in the first LSTM or GRU layer.
    - dropout_rate: float, dropout rate to prevent overfitting during training.
    - optimizer: str, the optimizer for model training ('adam', 'rmsprop', etc.).
    - sequence_array: np.array, the input data sequences (time-series data).
    - label_array: np.array, the output labels for training (binary labels for classification).

    Returns:
    - model: Keras Sequential model, compiled and ready for training.
    
    This function constructs the model architecture using either LSTM or GRU layers 
    based on the provided `model_type`. Each model consists of two recurrent layers 
    followed by dropout layers to reduce overfitting. A dense output layer is added 
    with sigmoid activation for binary classification (remaining useful life prediction).
    """
    model = Sequential()

    if model_type == 'LSTM':
        # LSTM layers with dropout to avoid overfitting
        # Input layer for the LSTM model (shape is based on sequence length and features)
        model.add(Input(shape=(sequence_array.shape[1], sequence_array.shape[2])))
        model.add(LSTM(units=units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=int(units/2), return_sequences=False)) # Second LSTM layer with half the units
        model.add(Dropout(dropout_rate))
    elif model_type == 'GRU':
        # GRU layers with dropout to avoid overfitting
        # Input layer for the GRU model (shape is based on sequence length and features)
        model.add(Input(shape=(sequence_array.shape[1], sequence_array.shape[2])))
        model.add(GRU(units=units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(GRU(units=int(units/2), return_sequences=False)) # Second GRU layer with half the units
        model.add(Dropout(dropout_rate))
    else:
        raise ValueError("Unsupported model type. Choose either 'LSTM' or 'GRU'.")

    # Output layer with sigmoid activation for binary classification (RUL prediction)
    model.add(Dense(units=label_array.shape[1], activation='sigmoid'))
    
    # Compile the model with binary cross-entropy loss and the chosen optimizer
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def train_model(model_type, fd, model, sequence_array, label_array, model_path):
    """
    Train the LSTM or GRU model with the provided training data.

    Parameters:
    - model_type: str, either 'LSTM' or 'GRU'.
    - fd: str, the dataset identifier (e.g., 'FD001').
    - model: Keras model, the compiled model to train.
    - sequence_array: np.array, the input sequences for training.
    - label_array: np.array, the output labels for training.
    - model_path: str, path to save the best model during training.

    Returns:
    - history: Keras History object, containing details of the training process.

    This function trains the selected LSTM or GRU model using the provided data.
    It implements early stopping to prevent overfitting and saves the best model based on validation loss.
    """
    print(f"\nTraining {model_type} model for dataset {fd} with units=50, dropout_rate=0.2, batch_size=32, epochs=100, optimizer=adam")

    # Start the timer to measure training time
    start_time = time.time()

    # Train the model with early stopping and model checkpointing
    history = model.fit(
        sequence_array,
        label_array,
        epochs=100, # Train for up to 100 epochs
        batch_size=32, # Use a batch size of 32
        validation_split=0.2, # Use 20% of the data as validation set
        verbose=0, # Suppress verbose output during training
        callbacks=[
            # Early stopping to prevent overfitting, stop if validation loss doesn't improve for 10 epochs
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=10,
                verbose=1,
                mode='min'
            ),
            # Save the best model based on the validation loss
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=0
            )
        ]
    )

    # End the timer and calculate the time spent
    end_time = time.time()
    time_spent = (end_time - start_time) / 60

    # Print the time taken for training
    print(f"Time spent training {model_type} model for dataset {fd}: {time_spent:.2f} minutes")

    return history