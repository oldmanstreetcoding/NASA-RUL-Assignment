# Import necessary libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Function to train and evaluate the model
def evaluate_model_performance(history, model, model_type, sequence_array, label_array, fd):

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'Model Accuracy - {model_type} for {fd}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'visualizations/{fd}_accuracy_plot_{model_type.lower()}.png')

    # Plot loss
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss - {model_type} for {fd}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'visualizations/{fd}_loss_plot_{model_type.lower()}.png')


    print(f"Model Performance on Training Data for The Best {model_type} Model in {fd}")
    
    # Evaluate model on training data
    scores = model.evaluate(sequence_array, label_array, verbose=0)
    print(f'>> Accuracy : {scores[1] * 100:.2f}%')

    # Make predictions
    y_pred_prob = model.predict(sequence_array)
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = label_array

    # Save predictions to CSV
    test_set = pd.DataFrame({'Actual': y_true.flatten(), 'Predicted': y_pred.flatten()})
    test_set.to_csv(f'data/csv/{fd}_predictions_{model_type.lower()}.csv', index=False)

    # Compute precision, recall and f1 score
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f'>> Precision ({model_type}) = {precision * 100:.2f}% \n>> Recall ({model_type}) = {recall * 100:.2f}%\n>> F1-score ({model_type}) = {f1 * 100:.2f}%')

    # Compute confusion matrix
    print('>> Confusion matrix:')
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print('')

    return scores, precision, recall, f1

# Function to evaluate models on the test data
def evaluate_test_model(test_df, sequence_columns, model_path, sequence_length, fd, model_type):
    # Prepare test data
    seq_array_test_last = [test_df[test_df['id'] == id][sequence_columns].values[-sequence_length:] 
                           for id in test_df['id'].unique() if len(test_df[test_df['id'] == id]) >= sequence_length]
    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

    # Pick the labels
    y_mask = [len(test_df[test_df['id'] == id]) >= sequence_length for id in test_df['id'].unique()]
    label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)

    # Load the saved model
    if os.path.isfile(model_path):
        estimator = load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file {model_path} not found. Ensure that the model is trained and saved properly.")
    
    print(f"Model Performance on Test Data for The Best {model_type} Model in {fd}")

    # Evaluate the model
    scores_test = estimator.evaluate(seq_array_test_last, label_array_test_last, verbose=0)
    print(f'>> Accuracy: {scores_test[1] * 100:.2f}%')

    # Make predictions
    y_pred_test_prob = estimator.predict(seq_array_test_last)
    y_pred_test = (y_pred_test_prob > 0.5).astype(int)
    y_true_test = label_array_test_last

    # Save predictions
    test_set = pd.DataFrame({'Actual': y_true_test.flatten(), 'Predicted': y_pred_test.flatten()})
    test_set.to_csv(f'data/csv/{fd}_est_predictions_{model_path.split("_")[-1].split(".")[0].lower()}.csv', index=False)

    # Compute precision, recall, and F1-score
    precision_test = precision_score(y_true_test, y_pred_test)
    recall_test = recall_score(y_true_test, y_pred_test)
    f1_test = f1_score(y_true_test, y_pred_test)
    print(f'>> Precision ({model_type}) = {precision_test * 100:.2f}% \n>> Recall ({model_type}) = {recall_test * 100:.2f}%\n>> F1-score ({model_type}) = {f1_test * 100:.2f}%')

    # Compute Confusion Matrix
    cm = confusion_matrix(y_true_test, y_pred_test)
    print('>> Confusion matrix:')
    print(cm)
    print('')

    # Plot actual vs predicted data
    plt.figure(figsize=(10, 6))
    plt.plot(y_true_test, label='Actual')
    plt.plot(y_pred_test, label='Predicted')
    plt.title(f'Actual vs Predicted RUL - {model_path.split("_")[-1].split(".")[0].upper()}')
    plt.xlabel('Samples')
    plt.ylabel('Remaining Useful Life')
    plt.legend()
    plt.savefig(f'visualizations/{fd}_actual_vs_predicted_{model_path.split("_")[-1].split(".")[0].lower()}.png')

    return scores_test, precision_test, recall_test, f1_test

# Function to create comparison plots for LSTM and GRU models
def evaluate_comparison_plot(results_df, fd):
    results_df.plot(kind='bar', figsize=(10, 6))
    plt.title(f'Model Performance Comparison for {fd} : LSTM vs GRU')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'visualizations/{fd}_model_comparison_bar_chart.png')

# Function to perform Monte Carlo Dropout and compute prediction intervals
def monte_carlo_dropout_prediction(model, data, n_iterations=100, dropout_rate=0.2):
    # Generate multiple predictions
    predictions = np.array([model.predict(data, verbose=0) for _ in range(n_iterations)])
    
    # Compute mean and standard deviation of predictions
    mean_predictions = predictions.mean(axis=0)
    std_predictions = predictions.std(axis=0)
    
    # Calculate prediction intervals (95% prediction interval assuming normal distribution)
    lower_bound = mean_predictions - 1.96 * std_predictions
    upper_bound = mean_predictions + 1.96 * std_predictions
    
    return mean_predictions, lower_bound, upper_bound

# Function to evaluate prediction with confidence interval
def evaluate_with_prediction_intervals(test_df, sequence_columns, model_path, sequence_length, fd, model_type):
    
    # Prepare test data
    seq_array_test_last = [test_df[test_df['id'] == id][sequence_columns].values[-sequence_length:] 
                           for id in test_df['id'].unique() if len(test_df[test_df['id'] == id]) >= sequence_length]
    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)

    # Pick the labels
    y_mask = [len(test_df[test_df['id'] == id]) >= sequence_length for id in test_df['id'].unique()]
    label_array_test_last = test_df.groupby('id')['label1'].nth(-1)[y_mask].values
    label_array_test_last = label_array_test_last.reshape(label_array_test_last.shape[0], 1).astype(np.float32)

    # Load the saved model
    if os.path.isfile(model_path):
        model = load_model(model_path)
    else:
        raise FileNotFoundError(f"Model file {model_path} not found. Ensure that the model is trained and saved properly.")

    print(f"Model Performance on Test Data with Confidence Interval for The Best {model_type} Model in {fd}")

    # Evaluate the model
    scores_test = model.evaluate(seq_array_test_last, label_array_test_last, verbose=0)
    print(f'>> Accuracy: {scores_test[1] * 100:.2f}%')

    # Make predictions with prediction intervals
    y_pred_mean, y_pred_lower, y_pred_upper = monte_carlo_dropout_prediction(model, seq_array_test_last, n_iterations=100)

    y_true_test = label_array_test_last

    # Save predictions to CSV
    test_set = pd.DataFrame({
        'Actual': y_true_test.flatten(),
        'Predicted Mean': y_pred_mean.flatten(),
        'Lower Bound': y_pred_lower.flatten(),
        'Upper Bound': y_pred_upper.flatten()
    })
    test_set.to_csv(f'data/csv/{fd}_test_predictions_{model_path.split("_")[-1].split(".")[0].lower()}_intervals.csv', index=False)

    # Compute precision, recall, and F1-score
    y_pred_test = (y_pred_mean > 0.5).astype(int)  # Use mean predictions for threshold-based classification
    precision_test = precision_score(y_true_test, y_pred_test)
    recall_test = recall_score(y_true_test, y_pred_test)
    f1_test = f1_score(y_true_test, y_pred_test)
    print(f'>> Precision ({model_type}) = {precision_test * 100:.2f}% \n>> Recall ({model_type}) = {recall_test * 100:.2f}%\n>> F1-score ({model_type}) = {f1_test * 100:.2f}%')

    # Compute Confusion Matrix
    cm = confusion_matrix(y_true_test, y_pred_test)
    print('Confusion matrix:')
    print(cm)
    print('')

    # Plot actual vs predicted data with prediction intervals
    plt.figure(figsize=(10, 6))
    plt.plot(y_true_test, label='Actual', color='blue')
    plt.plot(y_pred_mean, label='Predicted Mean', color='green')
    plt.fill_between(range(len(y_pred_mean)), y_pred_lower.flatten(), y_pred_upper.flatten(), color='orange', alpha=0.3, label='95% Prediction Interval')
    plt.title(f'Actual vs Predicted RUL with Prediction Intervals - {model_path.split("_")[-1].split(".")[0].upper()}')
    plt.xlabel('Samples')
    plt.ylabel('Remaining Useful Life')
    plt.legend()
    plt.savefig(f'visualizations/{fd}_actual_vs_predicted_{model_path.split("_")[-1].split(".")[0].lower()}_prediction_intervals.png')

    return scores_test, precision_test, recall_test, f1_test, cm