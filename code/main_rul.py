# Import necessary modules
import pandas as pd
import numpy as np

from data_preprocessing import load_data, preprocess_data
from model_building import create_model, train_model
from evaluation import evaluate_model_performance, evaluate_test_model, evaluate_comparison_plot, evaluate_with_prediction_intervals
from hyperparameter_tuning import execute_hyperparameter_search

# Setting seed for reproducibility
np.random.seed(1234)
PYTHONHASHSEED = 0

def display_menu():
    print("\n== Turbofan Engine Maintenance Predictor ==")
    print("Select Which Dataset You Want to Load:")
    print("1. FD001")
    print("2. FD001 + Tuning")
    print("3. FD002")
    print("4. FD002 + Tuning")
    print("5. FD001 & FD002")
    print("6. FD001 & FD002 + Tuning")
    print("7. Exit")
    print("== Turbofan Engine Maintenance Predictor ==")

def main():
    display_menu()
    option = input("\nEnter your choice (1/2/3/4/5/6/7): ")

    if option == '7':  # Exit option
        print("\nExiting the program. Goodbye!\n")
    
    # Dictionary to store evaluation inputs and paths for comparison
    evaluation_data = {}

    if option in ['1', '2', '3', '4', '5', '6']:

        if option == '1' or option == '2':
            fd_name = ["FD001"]
        elif option == '3' or option == '4': 
            fd_name = ["FD002"]
        elif option == '5' or option == '6': 
            fd_name = ["FD001", "FD002"]

        if option == '1' or option == '3' or option == '5':
            tuning = ''
        elif option == '2' or option == '4' or option == '6': 
            tuning = 'with Hyperparameter Tuning'
        
        for fd in fd_name:

            print(f"\nProcessing {fd} dataset {tuning}...\n")

            # Define file paths for training, testing, and truth datasets
            train_path = f'data/train_{fd}.txt'
            test_path = f'data/test_{fd}.txt'
            truth_path = f'data/RUL_{fd}.txt'

            # Data loading
            train_df, test_df, truth_df = load_data(train_path, test_path, truth_path)
        
            # Preprocess data
            train_df, test_df, sequence_length, sequence_columns, sequence_array, label_array = preprocess_data(train_df, test_df, truth_df)

            best_model_lstm_path = f'data/model/{fd}_the_best_lstm.keras'
            best_model_gru_path = f'data/model/{fd}_the_best_gru.keras'

            list_models = ['LSTM', 'GRU']

            if tuning == '':
                # Without hyperparameter tuning, train both LSTM and GRU models
                for model_type in list_models:
                    if model_type == 'LSTM':
                        model_path = best_model_lstm_path
                    elif model_type == 'GRU':
                        model_path = best_model_gru_path

                    model = create_model(model_type, 50, 0.2, 'adam', sequence_array, label_array)
                    history = train_model(model_type, fd, model, sequence_array, label_array, model_path)
                    evaluate_model_performance(history, model, model_type, sequence_array, label_array, fd)

            else:
                # Perform hyperparameter tuning
                print(f"Starting hyperparameter tuning for {fd} dataset...\n")
                execute_hyperparameter_search(sequence_array, label_array, fd, best_model_lstm_path, best_model_gru_path)

            # Store test data and model paths for evaluation later
            evaluation_data[fd] = {
                'test_df': test_df,
                'sequence_columns': sequence_columns,
                'sequence_length': sequence_length,
                'best_model_lstm_path': best_model_lstm_path,
                'best_model_gru_path': best_model_gru_path
            }

        
    # *** Evaluate after the loop ***
    for fd in evaluation_data:
        print(f"\nEvaluating models for {fd} dataset...\n")

        test_df = evaluation_data[fd]['test_df']
        sequence_columns = evaluation_data[fd]['sequence_columns']
        sequence_length = evaluation_data[fd]['sequence_length']
        best_model_lstm_path = evaluation_data[fd]['best_model_lstm_path']
        best_model_gru_path = evaluation_data[fd]['best_model_gru_path']

        # Evaluate LSTM model on the test data
        lstm_test_scores, lstm_test_precision, lstm_test_recall, lstm_test_f1 = evaluate_test_model(
            test_df, sequence_columns, best_model_lstm_path, sequence_length, fd, 'LSTM')

        # Evaluate GRU model on the test data
        gru_test_scores, gru_test_precision, gru_test_recall, gru_test_f1 = evaluate_test_model(
            test_df, sequence_columns, best_model_gru_path, sequence_length, fd, 'GRU')

        # Compare results between LSTM, GRU, and the template best model
        results_df = pd.DataFrame([
            [lstm_test_scores[1], lstm_test_precision, lstm_test_recall, lstm_test_f1],
            [gru_test_scores[1], gru_test_precision, gru_test_recall, gru_test_f1],
            [0.94, 0.952381, 0.8, 0.869565]  # Template Best Model
        ], columns=['Accuracy', 'Precision', 'Recall', 'F1-score'],
        index=['LSTM', 'GRU', 'Template Best Model'])

        print(results_df)
        print('')

        # Plot the comparison of results
        evaluate_comparison_plot(results_df, fd)

        # Evaluate LSTM model with prediction intervals
        evaluate_with_prediction_intervals(test_df, sequence_columns, best_model_lstm_path, sequence_length, fd, 'LSTM')

        # Evaluate GRU model with prediction intervals
        evaluate_with_prediction_intervals(test_df, sequence_columns, best_model_gru_path, sequence_length, fd, 'GRU')

if __name__ == '__main__':
    main()
