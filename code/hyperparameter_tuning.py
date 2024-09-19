import pandas as pd
import matplotlib.pyplot as plt
import time

from model_building import create_model
from evaluation import evaluate_model_performance

from keras.callbacks import EarlyStopping
from sklearn.model_selection import ParameterGrid

def execute_hyperparameter_search(sequence_array, label_array, fd, best_model_lstm_path, best_model_gru_path):
    """
    Perform hyperparameter search, save the best model, and visualize the results.

    Parameters:
    - sequence_array: numpy array, input sequences for training.
    - label_array: numpy array, corresponding labels for training.
    """
    # Define the grid of hyperparameters
    param_grid = {
        'model_type': ['LSTM', 'GRU'],
        'units': [32, 50, 64, 100],
        'dropout_rate': [0.1, 0.2, 0.3, 0.4],
        'batch_size': [16, 32, 64],
        'optimizer': ['adam', 'rmsprop']
    }

    # Initialize variables to track the best models
    results = []
    best_val_accuracy_lstm = 0  # Best validation accuracy for LSTM
    best_val_accuracy_gru = 0   # Best validation accuracy for GRU
    smallest_gap_lstm = float('inf')  # Smallest accuracy gap for LSTM
    smallest_gap_gru = float('inf')   # Smallest accuracy gap for GRU
    best_model_lstm = None  # Best LSTM model
    best_model_gru = None   # Best GRU model

    # Perform grid search
    for params in ParameterGrid(param_grid):
        model, train_acc, val_acc, accuracy_gap, history, time_spent = train_hyperparameter_tuning(
            model_type=params['model_type'],
            units=params['units'],
            dropout_rate=params['dropout_rate'],
            batch_size=params['batch_size'],
            optimizer=params['optimizer'],
            sequence_array=sequence_array,
            label_array=label_array
        )

        # Append results to the results list
        results.append({
            'Dataset': fd,
            'Model_Type': params['model_type'],
            'Units': params['units'],
            'Dropout_Rate': params['dropout_rate'],
            'Batch_Size': params['batch_size'],
            'Optimizer': params['optimizer'],
            'Train_Accuracy': train_acc,
            'Val_Accuracy': val_acc,
            'Accuracy_Gap': accuracy_gap,
            'Train_Time': time_spent
        })

        # Update the best LSTM model if applicable
        if params['model_type'] == 'LSTM' and (
            val_acc > best_val_accuracy_lstm or 
            (val_acc == best_val_accuracy_lstm and accuracy_gap < smallest_gap_lstm)
        ):
            best_val_accuracy_lstm = val_acc
            smallest_gap_lstm = accuracy_gap
            
            best_model_lstm = model
            best_history_lstm = history

        # Update the best GRU model if applicable
        elif params['model_type'] == 'GRU' and (
            val_acc > best_val_accuracy_gru or 
            (val_acc == best_val_accuracy_gru and accuracy_gap < smallest_gap_gru)
        ):
            best_val_accuracy_gru = val_acc
            smallest_gap_gru = accuracy_gap
            
            best_model_gru = model
            best_history_gru = history

    # Convert results to DataFrame and sort
    results_df = pd.DataFrame(results)
    results_df.sort_values(by=['Dataset', 'Model_Type', 'Val_Accuracy', 'Accuracy_Gap'], ascending=[False, True, False, True], inplace=True)

    # Save results to CSV
    results_df.to_csv(f'data/csv/{fd}_hyperparameter_results.csv', index=False)

    print(results_df)

    # Save the best LSTM and GRU models
    if best_model_lstm:
        best_model_lstm.save(best_model_lstm_path)
        print(f"\nBest LSTM model for {fd} saved with validation accuracy: {best_val_accuracy_lstm * 100:.2f}% and accuracy gap: {smallest_gap_lstm * 100:.2f}%")

    if best_model_gru:
        best_model_gru.save(best_model_gru_path)
        print(f"Best GRU model for {fd} saved with validation accuracy: {best_val_accuracy_gru * 100:.2f}% and accuracy gap: {smallest_gap_gru * 100:.2f}%")

    print(f"\nHyperparameter tuning complete for {fd}. Results saved to 'data/csv/{fd}_hyperparameter_results.csv'.\n")

    # Visualize the results
    visualize_hyperparameter_results(results_df, fd)

    evaluate_model_performance(best_history_lstm, best_model_lstm, 'LSTM', sequence_array, label_array, fd)

    evaluate_model_performance(best_history_gru, best_model_gru, 'GRU', sequence_array, label_array, fd)

def train_hyperparameter_tuning(model_type='LSTM', units=50, dropout_rate=0.2, batch_size=32, epochs=100, optimizer='adam', sequence_array=None, label_array=None, fd='FD001'):
    """
    Train the model with specified hyperparameters.

    Parameters:
    - model_type: str, either 'LSTM' or 'GRU'.
    - units: int, number of units in each recurrent layer.
    - dropout_rate: float, dropout rate to prevent overfitting.
    - batch_size: int, batch size for training.
    - epochs: int, number of epochs for training.
    - optimizer: str, optimizer to use ('adam', 'rmsprop', etc.).
    - sequence_array: numpy array, input sequences for training.
    - label_array: numpy array, corresponding labels for training.

    Returns:
    - model: trained Keras model.
    - train_accuracy: float, final training accuracy.
    - val_accuracy: float, final validation accuracy.
    - accuracy_gap: float, absolute gap between training and validation accuracy.
    """
    
    # Create model
    model = create_model(model_type, units, dropout_rate, optimizer, sequence_array, label_array)

    print(f"\nTraining {model_type} model for dataset {fd} with units={units}, dropout_rate={dropout_rate}, batch_size={batch_size}, epochs={epochs}, optimizer={optimizer}")

    # Start the timer
    start_time = time.time()

    # Define EarlyStopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0.001,
        patience=10,
        verbose=1,
        mode='min'
    )

    # Train the model with EarlyStopping
    history = model.fit(
        sequence_array, 
        label_array, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_split=0.2, 
        verbose=0, 
        callbacks=[early_stopping]
    )

    # End the timer
    end_time = time.time()

    # Calculate the time spent (in minutes)
    time_spent = (end_time - start_time) / 60

    # Print the time spent
    print(f"Time spent training {model_type} model for dataset {fd}: {time_spent:.2f} minutes")

    # Evaluate the model
    train_accuracy = history.history['accuracy'][-1]
    val_accuracy = history.history['val_accuracy'][-1]
    accuracy_gap = abs(train_accuracy - val_accuracy)  # Calculate the gap between training and validation accuracy
    print(f'>> Training Accuracy: {train_accuracy * 100:.2f}%')
    print(f'>> Validation Accuracy: {val_accuracy * 100:.2f}%')
    print(f'>> Accuracy Gap: {accuracy_gap * 100:.2f}%')

    return model, train_accuracy, val_accuracy, accuracy_gap, history, time_spent

def visualize_hyperparameter_results(results_df, fd):
    """
    Visualize the hyperparameter tuning results.

    Parameters:
    - results_df: pandas DataFrame, containing the hyperparameter tuning results.
    """
    
    # Sorting by model type for better visualization
    results_df.sort_values(by=['Dataset', 'Model_Type', 'Val_Accuracy', 'Accuracy_Gap'], ascending=[False, True, False, True], inplace=True)
    
    # Create subplots for comparison of parameters
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Hyperparameters vs Validation Accuracy for LSTM and GRU Models for {fd}', fontsize=16)

    # Units vs Validation Accuracy
    for model_type in results_df['Model_Type'].unique():
        subset = results_df[results_df['Model_Type'] == model_type]
        axes[0, 0].plot(subset['Units'], subset['Val_Accuracy'], marker='o', linestyle='-', label=f'{model_type}')
    axes[0, 0].set_title('Units vs Validation Accuracy')
    axes[0, 0].set_xlabel('Units')
    axes[0, 0].set_ylabel('Validation Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Dropout Rate vs Validation Accuracy
    for model_type in results_df['Model_Type'].unique():
        subset = results_df[results_df['Model_Type'] == model_type]
        axes[0, 1].plot(subset['Dropout_Rate'], subset['Val_Accuracy'], marker='o', linestyle='-', label=f'{model_type}')
    axes[0, 1].set_title('Dropout Rate vs Validation Accuracy')
    axes[0, 1].set_xlabel('Dropout Rate')
    axes[0, 1].set_ylabel('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Batch Size vs Validation Accuracy
    for model_type in results_df['Model_Type'].unique():
        subset = results_df[results_df['Model_Type'] == model_type]
        axes[1, 0].plot(subset['Batch_Size'], subset['Val_Accuracy'], marker='o', linestyle='-', label=f'{model_type}')
    axes[1, 0].set_title('Batch Size vs Validation Accuracy')
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('Validation Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Optimizer vs Validation Accuracy
    for model_type in results_df['Model_Type'].unique():
        subset = results_df[results_df['Model_Type'] == model_type]
        axes[1, 1].plot(subset['Optimizer'], subset['Val_Accuracy'], marker='o', linestyle='-', label=f'{model_type}')
    axes[1, 1].set_title('Optimizer vs Validation Accuracy')
    axes[1, 1].set_xlabel('Optimizer')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'visualizations/{fd}_hyperparameters_vs_val_accuracy.png')

    # Plotting the comparison chart for validation accuracy vs accuracy gap
    plt.figure(figsize=(15, 8))
    for model_type in results_df['Model_Type'].unique():
        subset = results_df[results_df['Model_Type'] == model_type]
        plt.scatter(subset['Val_Accuracy'], subset['Accuracy_Gap'], label=f'{model_type} Model', s=100)

    plt.title(f'Validation Accuracy vs Accuracy Gap for LSTM and GRU Models in {fd}')
    plt.xlabel('Validation Accuracy')
    plt.ylabel('Accuracy Gap (Train - Validation)')
    plt.legend(loc='best')
    plt.grid(True)

    # Save plot
    plt.savefig(f'visualizations/{fd}_val_accuracy_vs_gap.png')