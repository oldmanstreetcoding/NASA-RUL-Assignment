# Import necessary functions
from data_loading_preprocessing import load_and_preprocess_data
from model_building import build_and_train_model

# Load and preprocess the data
PM_train = 'CMAPSSData/train_FD001.txt'
PM_test = 'CMAPSSData/test_FD001.txt'
PM_truth = 'CMAPSSData/RUL_FD001.txt'
train_df, test_df = load_and_preprocess_data(PM_train, PM_test, PM_truth)

# Build and train the initial model
model, history = build_and_train_model(train_df)