# Import necessary modules
import pandas as pd
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppressing TensorFlow floating-point warnings

# Import custom modules for different phases of the project
from data_preprocessing import load_data, preprocess_data  # For data loading and preprocessing
from model_building import create_model, train_model  # For LSTM/GRU model creation and training
from evaluation import evaluate_model_performance, evaluate_test_model, evaluate_comparison_plot, evaluate_with_prediction_intervals  # For evaluation
from hyperparameter_tuning import execute_hyperparameter_search  # For hyperparameter tuning

# Setting seed for reproducibility to ensure consistent model results across different runs
np.random.seed(1234)
PYTHONHASHSEED = 0

# Display menu for user interaction
def display_menu():
    """
    Function to display the available options for dataset selection and model configuration.
    Provides options to work with different datasets (FD001, FD002) and apply hyperparameter tuning.
    """
    print("\n== Turbofan Engine Maintenance Predictor ==")
    print("Select Which Dataset You Want to Load:")
    print("1. FD001")  # Option 1: FD001 with static parameters
    print("2. FD001 << Tuning")  # Option 2: FD001 with hyperparameter tuning
    print("3. FD002")  # Option 3: FD002 with static parameters
    print("4. FD002 << Tuning")  # Option 4: FD002 with hyperparameter tuning
    print("5. FD001 & FD002")  # Option 5: Both datasets with static parameters
    print("6. FD001 & FD002 << Tuning")  # Option 6: Both datasets with hyperparameter tuning
    print("7. FD001 << Evaluate (Pre-trained)")  # Option 7: Evaluate the pre-trained Best Model for FD001
    print("8. FD002 << Evaluate (Pre-trained)")  # Option 8: Evaluate the pre-trained Best Model for FD002
    print("9. FD001 & FD002 << Evaluate (Pre-trained)")  # Option 9: Evaluate the pre-trained Best Models for Both datasets
    print("10. Exit")  # Option to exit the program
    print("== Turbofan Engine Maintenance Predictor ==")

def main():
    """
    Main function for the engine maintenance predictor program.
    Controls the flow of the program by loading data, training models, and evaluating their performance.
    """
    display_menu()
    option = input("\nEnter your choice (1-10): ")

    if option == '10':  # Exit option
        print("\nExiting the program. Goodbye!\n")
        return
    
    # Dictionary to store evaluation inputs and paths for comparison between LSTM and GRU models
    evaluation_data = {}

    # Define the options where the models should be evaluated directly without retraining
    evaluate_only = option in ['7', '8', '9']

    # Determine which dataset(s) to process
    if option in ['1', '2', '7']:
        fd_name = ["FD001"]
    elif option in ['3', '4', '8']:
        fd_name = ["FD002"]
    elif option in ['5', '6', '9']:
        fd_name = ["FD001", "FD002"]

    # Determine if hyperparameter tuning should be applied
    if option in ['1', '3', '5', '7', '8', '9']:
        tuning = ''
    else:
        tuning = 'with Hyperparameter Tuning'

    # Loop through each selected dataset (FD001, FD002)
    for fd in fd_name:

        print(f"\nProcessing {fd} dataset {tuning}...\n")

        # === Part 1: Data Acquisition and Preprocessing (20%) ===
        # Define file paths for training, testing, and truth datasets
        train_path = f'data/train_{fd}.txt'  # Path to training data
        test_path = f'data/test_{fd}.txt'  # Path to testing data
        truth_path = f'data/RUL_{fd}.txt'  # Path to Remaining Useful Life (RUL) ground truth

        # Data loading from NASA CMAPSS dataset
        train_df, test_df, truth_df = load_data(train_path, test_path, truth_path)
    
        # Preprocess data, including normalization and feature extraction (can be extended with additional techniques)
        train_df, test_df, sequence_length, sequence_columns, sequence_array, label_array = preprocess_data(train_df, test_df, truth_df)

        # Define model save paths for both LSTM and GRU models
        best_model_lstm_path = f'data/model/{fd}_the_best_lstm.keras'
        best_model_gru_path = f'data/model/{fd}_the_best_gru.keras'

        # === Part 2: Model Building (30%) ===
        if not evaluate_only:
            list_models = ['LSTM', 'GRU']  # List of models to train

            if tuning == '':  # If no hyperparameter tuning is selected
                # Train both LSTM and GRU models on the preprocessed data
                for model_type in list_models:
                    if model_type == 'LSTM':
                        model_path = best_model_lstm_path
                    elif model_type == 'GRU':
                        model_path = best_model_gru_path

                    # Create and train the selected model (LSTM or GRU)
                    model = create_model(model_type, 50, 0.2, 'adam', sequence_array, label_array)
                    history = train_model(model_type, fd, model, sequence_array, label_array, model_path)

                    # Evaluate the model's performance on training data and validation split
                    evaluate_model_performance(history, model, model_type, sequence_array, label_array, fd)

            else:  # If hyperparameter tuning is selected
                # Perform hyperparameter tuning to optimize LSTM and GRU models
                print(f"Starting hyperparameter tuning for {fd} dataset...\n")
                execute_hyperparameter_search(sequence_array, label_array, fd, best_model_lstm_path, best_model_gru_path)

        # Store test data and model paths for evaluation after training or direct evaluation
        evaluation_data[fd] = {
            'test_df': test_df,
            'sequence_columns': sequence_columns,
            'sequence_length': sequence_length,
            'best_model_lstm_path': best_model_lstm_path,
            'best_model_gru_path': best_model_gru_path
        }

    # === Part 3: Model Evaluation and Comparison (25%) ===
    # Evaluate the performance of LSTM and GRU models after training or direct evaluation
    for fd in evaluation_data:
        print(f"\nEvaluating models for {fd} dataset...\n")

        test_df = evaluation_data[fd]['test_df']  # Get test data for evaluation
        sequence_columns = evaluation_data[fd]['sequence_columns']  # Feature columns used for evaluation
        sequence_length = evaluation_data[fd]['sequence_length']  # Sequence length for time series
        best_model_lstm_path = evaluation_data[fd]['best_model_lstm_path']  # Path to the best LSTM model
        best_model_gru_path = evaluation_data[fd]['best_model_gru_path']  # Path to the best GRU model

        # Evaluate LSTM model on the test data
        lstm_test_scores, lstm_test_precision, lstm_test_recall, lstm_test_f1 = evaluate_test_model(
            test_df, sequence_columns, best_model_lstm_path, sequence_length, fd, 'LSTM')

        # Evaluate GRU model on the test data
        gru_test_scores, gru_test_precision, gru_test_recall, gru_test_f1 = evaluate_test_model(
            test_df, sequence_columns, best_model_gru_path, sequence_length, fd, 'GRU')

        # Compare results between LSTM, GRU, and a template best model (hypothetical benchmark)
        results_df = pd.DataFrame([
            [lstm_test_scores[1], lstm_test_precision, lstm_test_recall, lstm_test_f1],
            [gru_test_scores[1], gru_test_precision, gru_test_recall, gru_test_f1],
            [0.94, 0.952381, 0.8, 0.869565]  # Template Best Model (as a benchmark)
        ], columns=['Accuracy', 'Precision', 'Recall', 'F1-score'],
        index=['LSTM', 'GRU', 'Template Best Model'])

        # Print and visualize the comparison results
        print(results_df)
        print('')

        # Save the DataFrame to a CSV file
        csv_output_path = f'data/csv/results_{fd}.csv'  # Define a file path
        results_df.to_csv(csv_output_path, index=True)
        print(f"Comparison Results saved to {csv_output_path}\n")

        # Plot the comparison of LSTM and GRU model performance
        evaluate_comparison_plot(results_df, fd)

        # Evaluate LSTM model with prediction intervals for more insightful visualization
        evaluate_with_prediction_intervals(test_df, sequence_columns, best_model_lstm_path, sequence_length, fd, 'LSTM')

        # Evaluate GRU model with prediction intervals
        evaluate_with_prediction_intervals(test_df, sequence_columns, best_model_gru_path, sequence_length, fd, 'GRU')

if __name__ == '__main__':
    main()