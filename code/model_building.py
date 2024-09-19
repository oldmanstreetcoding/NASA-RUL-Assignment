# Import necessary libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Function to create LSTM or GRU models
def create_model(model_type, units, dropout_rate, optimizer, sequence_array, label_array):

    model = Sequential()

    if model_type == 'LSTM':
        model.add(LSTM(units=units, input_shape=(sequence_array.shape[1], sequence_array.shape[2]), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=int(units/2), return_sequences=False))
        model.add(Dropout(dropout_rate))
    elif model_type == 'GRU':
        model.add(GRU(units=units, input_shape=(sequence_array.shape[1], sequence_array.shape[2]), return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(GRU(units=int(units/2), return_sequences=False))
        model.add(Dropout(dropout_rate))
    else:
        raise ValueError("Unsupported model type. Choose either 'LSTM' or 'GRU'.")

    model.add(Dense(units=label_array.shape[1], activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model

def train_model(model_type, fd, model, sequence_array, label_array, model_path):

    print(f"\nTraining {model_type} model for dataset {fd} with units=50, dropout_rate=0.2, batch_size=32, epochs=100, optimizer=adam")

    # Train model
    history = model.fit(
        sequence_array,
        label_array,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            # Conditional stopping
            EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=10,
                verbose=1,
                mode='min'
            ),
            # Save the best model
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=0
            )
        ]
    )

    return history