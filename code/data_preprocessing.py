# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Data loading
def load_data(train_path, test_path, truth_path):
    """
    Function to load the training, test, and truth datasets.

    Parameters:
    - train_path: str, path to the training data.
    - test_path: str, path to the test data.
    - truth_path: str, path to the RUL (Remaining Useful Life) truth data.

    Returns:
    - train_df: DataFrame containing the training data.
    - test_df: DataFrame containing the test data.
    - truth_df: DataFrame containing the truth RUL values.
    """
    train_df = pd.read_csv(train_path, sep=' ', header=None)
    test_df = pd.read_csv(test_path, sep=' ', header=None)
    truth_df = pd.read_csv(truth_path, sep=' ', header=None)
    return train_df, test_df, truth_df

def preprocess_data(train_df, test_df, truth_df):
    """
    Function to preprocess the training and test datasets:
    - Drop unnecessary columns
    - Handle missing values
    - Normalize sensor and operational settings data
    - Feature engineering: Calculate RUL, generate labels, and prepare sequences for LSTM/GRU models.

    Parameters:
    - train_df: DataFrame containing the training data.
    - test_df: DataFrame containing the test data.
    - truth_df: DataFrame containing the truth RUL values.

    Returns:
    - train_df: Processed training DataFrame.
    - test_df: Processed test DataFrame.
    - sequence_length: int, the window size used for sequence generation.
    - sequence_columns: list, columns used in sequence generation.
    - sequence_array: np.array, generated sequences from the training data.
    - label_array: np.array, labels corresponding to the sequences.
    """

    # Data cleaning: Dropping unnecessary columns (column 26 and 27 are irrelevant)
    # Column 26 and 27 are blank columns introduced by extra spaces in the dataset.
    train_df.drop([26, 27], axis=1, inplace=True)
    test_df.drop([26, 27], axis=1, inplace=True)
    truth_df.drop([1], axis=1, inplace=True)

    # Handle missing data by dropping rows with missing values (if any)
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    truth_df = truth_df.dropna()

    # Assigning column names for clarity and easier referencing
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]

    # Sorting data by id and cycle to ensure proper sequence generation
    train_df = train_df.sort_values(['id', 'cycle'])

    # --- Feature Engineering ---

    # Calculate the Remaining Useful Life (RUL) for the training data
    # RUL = Maximum cycle for each engine - current cycle
    rul = train_df.groupby('id')['cycle'].max().reset_index()
    rul.columns = ['id', 'max_cycle']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    train_df.drop('max_cycle', axis=1, inplace=True)

    # Generate binary classification labels for remaining useful life (label1 and label2)
    # label1: Binary classification whether RUL <= 30 (1 = True, 0 = False)
    # label2: Multi-class classification, classifying RUL <= 15 as '2', RUL <= 30 as '1', otherwise '0'
    w1 = 30  # Threshold for label1
    w0 = 15  # Threshold for label2
    train_df['label1'] = (train_df['RUL'] <= w1).astype(int)
    train_df['label2'] = train_df['label1'].copy()
    train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

    # --- Data Normalization ---

    # Initialize MinMaxScaler to normalize sensor data between 0 and 1
    min_max_scaler = MinMaxScaler()
    cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2', 'cycle_norm'])

    # Normalize the training data (excluding non-numeric columns like id, cycle, RUL, labels)
    train_df['cycle_norm'] = (train_df['cycle'] - train_df['cycle'].min()) / (train_df['cycle'].max() - train_df['cycle'].min())
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), columns=cols_normalize)
    join_df = train_df[['id', 'cycle', 'cycle_norm', 'RUL', 'label1', 'label2']].join(norm_train_df)
    train_df = join_df.reindex(columns=['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)] + ['RUL', 'label1', 'label2', 'cycle_norm'])

    # Normalize the test data in the same way using the scaler fit from the training data
    test_df['cycle_norm'] = (test_df['cycle'] - test_df['cycle'].min()) / (test_df['cycle'].max() - test_df['cycle'].min())
    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), columns=cols_normalize)
    test_join_df = test_df[['id', 'cycle', 'cycle_norm']].join(norm_test_df)
    test_df = test_join_df.reindex(columns=['id', 'cycle', 'cycle_norm'] + list(cols_normalize))
    test_df = test_df.reindex(columns=['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)] + ['cycle_norm'])

    # --- Calculate RUL for Test Data ---

    # Similar process for test data, calculating RUL based on truth dataset
    rul = test_df.groupby('id')['cycle'].max().reset_index()
    rul.columns = ['id', 'max']
    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more']
    truth_df.drop(['more'], axis=1, inplace=True)
    test_df = test_df.merge(truth_df[['id', 'max']], on='id')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)

    # Generate labels for test data similar to training data
    test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0)
    test_df['label2'] = test_df['label1'].copy()
    test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2
    test_df = test_df.reindex(columns=['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)] + ['RUL', 'label1', 'label2', 'cycle_norm'])

    # --- Sequence Generation for LSTM/GRU ---

    sequence_length = 50 # Define window size (sequence length) for LSTM/GRU

    # Function to generate sliding window sequences for time series data
    def generate_sequences(id_df, sequence_length, feature_columns):
        data_matrix = id_df[feature_columns].values
        num_elements = data_matrix.shape[0]
        for start, end in zip(range(0, num_elements - sequence_length), range(sequence_length, num_elements)):
            yield data_matrix[start:end, :]

    # Feature columns used in sequence generation (operational settings + sensor data)
    sensor_columns = [f'sensor{i}' for i in range(1, 22)]
    sequence_columns = ['setting1', 'setting2', 'setting3', 'cycle_norm'] + sensor_columns

    # Generate sequences for training data
    sequence_generator = (list(generate_sequences(train_df[train_df['id'] == id], sequence_length, sequence_columns)) for id in train_df['id'].unique())
    sequence_array = np.concatenate(list(sequence_generator), axis=0)

    # Function to generate corresponding labels for sequences
    def generate_labels(id_df, sequence_length, label_column):
        data_matrix = id_df[label_column].values
        num_elements = data_matrix.shape[0]
        return data_matrix[sequence_length:num_elements].reshape(-1, 1)

    # Generate labels for sequences
    label_generator = (generate_labels(train_df[train_df['id'] == id], sequence_length, 'label1') for id in train_df['id'].unique())
    label_array = np.concatenate(list(label_generator), axis=0)

    return train_df, test_df, sequence_length, sequence_columns, sequence_array, label_array 