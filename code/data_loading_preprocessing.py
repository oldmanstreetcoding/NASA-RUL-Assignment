# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Setting seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

def load_and_preprocess_data(train_path, test_path, truth_path):
    """
    Load and preprocess the data from the specified paths.
    
    Parameters:
    - train_path: Path to the training data file
    - test_path: Path to the test data file
    - truth_path: Path to the ground truth file
    
    Returns:
    - train_df: Preprocessed training DataFrame
    - test_df: Preprocessed test DataFrame
    """

    # Read training data - Aircraft engine run-to-failure data
    # The data are provided as text files with 26 columns of numbers, separated by spaces
    train_df = pd.read_csv(train_path, sep=' ', header=None) # Read the txt file, use appropriate separator and header
    train_df.drop([26, 27], axis=1, inplace=True)  # Explore the data on your own and remove unnecessary columns
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]  # Assign names to all the columns

    train_df = train_df.sort_values(['id', 'cycle'])  # Sort by id and cycle

    # Read test data - Aircraft engine operating data without failure events recorded
    test_df = pd.read_csv(test_path, sep=' ', header=None)  # Read the txt file, use appropriate separator and header
    test_df.drop([26, 27], axis=1, inplace=True)  # Explore the data on your own and remove unnecessary columns
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]  # Assign names to all the columns

    # Read ground truth data - True remaining cycles for each engine in testing data
    truth_df = pd.read_csv(truth_path, sep=' ', header=None) # Read the txt file, use appropriate separator and header
    truth_df.drop([1], axis=1, inplace=True)  # Explore the data on your own and remove unnecessary columns

    # Data Preprocessing

    #######
    # TRAIN
    #######
    # Data Labeling - generate column RUL (Remaining Useful Life or Time to Failure)

    # TODO: Calculate the maximum cycle value for each engine (id) and store it in a new DataFrame (rul)
    rul = train_df.groupby('id')['cycle'].max().reset_index()
    # TODO: Rename the columns in the rul DataFrame
    rul.columns = ['id', 'max_cycle']
    # TODO: Merge the rul DataFrame with the original train_df based on the 'id' column
    train_df = train_df.merge(rul, on=['id'], how='left')
    # TODO: Calculate the Remaining Useful Life (RUL) by subtracting the current cycle from the maximum cycle
    train_df['RUL'] = train_df['max_cycle'] - train_df['cycle']
    # TODO: Remove the temporary column used to calculate RUL
    train_df.drop('max_cycle', axis=1, inplace=True)

    # Generate label columns for training data
    # We will only make use of "label1" for binary classification,
    # while trying to answer the question: is a specific engine going to fail within w1 cycles?
    w1 = 30
    w0 = 15

    # TODO: Create a binary label ('label1') indicating if the engine will fail within w1 cycles (1) or not (0)
    train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0)  # Replace with the correct threshold value and label values
    # TODO: Initialize a second label ('label2') as a copy of 'label1'
    train_df['label2'] = train_df['label1'].copy()
    # TODO: Update 'label2' to indicate if the engine will fail within w0 cycles (2) or not (0/1)
    train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2 # Replace with the correct threshold value and label value


    # MinMax normalization (from 0 to 1)
    # TODO: Create a normalized version of the 'cycle' column (e.g., 'cycle_norm') using the original 'cycle' values
    train_df['cycle_norm'] = (train_df['cycle'] - train_df['cycle'].min()) / (train_df['cycle'].max() - train_df['cycle'].min())  # Replace with the correct normalization code
    # TODO: Select the columns to be normalized (all columns except 'id', 'cycle', 'RUL', 'label1', and 'label2')
    cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2', 'cycle_norm'])  # Replace with the correct column selection code
    # TODO: Initialize a MinMaxScaler object to scale values between 0 and 1
    min_max_scaler = MinMaxScaler()  # Replace with the correct scaler initialization code
    # TODO: Apply MinMaxScaler to the selected columns and create a new normalized DataFrame
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), columns=cols_normalize)  # Replace with the correct normalization code
    # TODO: Join the normalized DataFrame with the original DataFrame (excluding normalized columns)
    join_df = train_df[['id', 'cycle', 'cycle_norm', 'RUL', 'label1', 'label2']].join(norm_train_df)  # Replace with the correct join code
    # TODO: Reorder the columns in the joined DataFrame to match the original order
    # train_df = join_df.reindex(columns=['id', 'cycle', 'cycle_norm', 'RUL', 'label1', 'label2'] + list(cols_normalize))  # Replace with the correct reindexing code
    train_df = join_df.reindex(columns=['id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)] + ['RUL', 'label1', 'label2', 'cycle_norm'])

    ######
    # TEST
    ######
    # MinMax normalization (from 0 to 1)
    # TODO: Similar to the MinMax normalization done for Train, complete the code below.

    test_df['cycle_norm'] = (test_df['cycle'] - test_df['cycle'].min()) / (test_df['cycle'].max() - test_df['cycle'].min())
    norm_test_df = pd.DataFrame(min_max_scaler.fit_transform(test_df[cols_normalize]), columns=cols_normalize)
    test_join_df = test_df[['id', 'cycle', 'cycle_norm']].join(norm_test_df)
    test_df = test_join_df.reindex(columns=['id', 'cycle', 'cycle_norm'] + list(cols_normalize))
    # test_df = NotImplemented

    # We use the ground truth dataset to generate labels for the test data.
    # generate column max for test data
    # TODO: Calculate the maximum cycle value for each engine (id) in the test data and store it in a new DataFrame (rul)
    rul = test_df.groupby('id')['cycle'].max().reset_index()
    # TODO: Rename the columns in the rul DataFrame
    rul.columns = ['id', 'max']
    # TODO: Merge the rul DataFrame with the original test_df based on the 'id' column
    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more']
    # TODO: Remove the temporary column used to calculate RUL
    truth_df.drop(['more'], axis=1, inplace=True)

    # TODO: Merge the adjusted truth_df with the test_df to generate RUL values for test data
    test_df = test_df.merge(truth_df[['id', 'max']], on='id')
    # TODO: Calculate the Remaining Useful Life (RUL) by subtracting the current cycle from the maximum cycle
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    # TODO: Remove the temporary column used to calculate RUL
    test_df.drop('max', axis=1, inplace=True)

    # Generate binary label columns (label1 and label2) based on RUL values and thresholds w0 and w1
    # TODO: Similar to what you did in the train dataframe
    test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0)
    test_df['label2'] = test_df['label1'].copy()
    test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2

    return train_df, test_df