import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import keras

# Define window size and sequence length
sequence_length = 50  # Example: Use a sequence length of 50 time steps

def generate_sequences(id_df, sequence_length, feature_columns):
    """Generate sequences from a dataframe for a given id.
    Sequences that are under the sequence length will be considered.
    We can also pad the sequences in order to use shorter ones."""
    data_matrix = id_df[feature_columns].values
    num_elements = data_matrix.shape[0]

    for start, end in zip(range(0, num_elements - sequence_length), range(sequence_length, num_elements)):
        yield data_matrix[start:end, :]

def generate_labels(id_df, sequence_length, label_column):
    """Generate labels for a given id."""
    data_matrix = id_df[label_column].values
    num_elements = data_matrix.shape[0]
    return data_matrix[sequence_length:num_elements]

def build_and_train_model(train_df):
    # Select feature columns for sequence generation
    sensor_columns = [f'sensor{i}' for i in range(1, 22)]
    sequence_columns = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + sensor_columns

    # Generate sequences for all engine ids in the training data
    sequence_generator = (list(generate_sequences(train_df[train_df['id'] == id], sequence_length, sequence_columns)) 
                          for id in train_df['id'].unique())
    sequence_array = np.concatenate(list(sequence_generator), axis=0)

    # Generate labels for all engine ids in the training data
    label_generator = (generate_labels(train_df[train_df['id'] == id], sequence_length, 'label1') 
                       for id in train_df['id'].unique())
    label_array = np.concatenate(list(label_generator), axis=0)

    # Define the number of features and output units
    nb_features = sequence_array.shape[2]
    nb_out = 1  # For binary classification, the output unit is 1

    # Create a Sequential model
    model = Sequential()

    # Add LSTM layers and Dropout layers to the model
    model.add(LSTM(units=50, input_shape=(sequence_length, nb_features), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=25, return_sequences=False))
    model.add(Dropout(0.2))

    # Add a Dense output layer with sigmoid activation
    model.add(Dense(units=nb_out, activation='sigmoid'))

    # Compile the model with binary crossentropy loss and Adam optimizer
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Print the model summary
    model.summary()

    # Fit the network to the training data
    history = model.fit(
        sequence_array,
        label_array,
        epochs=100,  # Example number of epochs
        batch_size=32,  # Example batch size
        validation_split=0.2,  # Example validation split proportion
        verbose=1,  # Example verbosity level
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.001,
                patience=10,
                verbose=1,
                mode='min'
            ),
            keras.callbacks.ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                verbose=1
            )
        ]
    )
    return model, history