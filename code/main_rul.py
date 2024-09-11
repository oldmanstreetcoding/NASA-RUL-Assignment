from data_collection_preprocess import load_and_preprocess_data

# Define file paths
PM_train = 'CMAPSSData/train_FD001.txt'
PM_test = 'CMAPSSData/test_FD001.txt'
PM_truth = 'CMAPSSData/RUL_FD001.txt'

# Load and preprocess the data
train_df, test_df = load_and_preprocess_data(PM_train, PM_test, PM_truth)

# Display the first few rows of the processed data
print("Training Data Sample:")
print(train_df.head())

print("\nTest Data Sample:")
print(test_df.head())

# Proceed with further analysis or model building using train_df and test_df