
# Predictive Maintenance Using LSTM and GRU on NASA Turbofan Engine

## Overview
This project focuses on **Predictive Maintenance** by predicting the **Remaining Useful Life (RUL)** of aircraft engines. Using **Long Short-Term Memory (LSTM)** networks and **Gated Recurrent Units (GRU)**, we analyze the data from the **NASA CMAPSS Turbofan Engine dataset** to develop models that predict engine failures before they occur. 

The project involves:
- Implementing LSTM and GRU models
- Hyperparameter tuning for optimal performance
- Comparing the model performance on FD001 and FD002 datasets

## Objective
The objective is to predict the Remaining Useful Life (RUL) of engines using LSTM and GRU models and to compare their performances on the **FD001** and **FD002** datasets from NASA's CMAPSS dataset.

---

## Dataset Information

**Dataset Source**: [NASA CMAPSS Jet Engine Simulated Data](https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6/about_data)

The dataset consists of multivariate time series data from a fleet of engines. Each engine operates under varying initial conditions and wears differently over time.

### FD001:
- **Train Trajectories**: 100
- **Test Trajectories**: 100
- **Conditions**: One (Sea Level)
- **Fault Modes**: One (HPC Degradation)

### FD002:
- **Train Trajectories**: 260
- **Test Trajectories**: 259
- **Conditions**: Six 
- **Fault Modes**: One (HPC Degradation)

### Dataset Format:
The dataset contains 26 columns for each operational cycle snapshot:
1. Unit number
2. Time, in cycles
3. Operational setting 1
4. Operational setting 2
5. Operational setting 3
6. Sensor measurement 1
...
26. Sensor measurement 26

### Data Preprocessing:
- Handling missing data
- Normalization using **MinMaxScaler**
- Feature engineering, including **Remaining Useful Life (RUL)** calculation and label generation for training and testing.
  
---

## Project Structure

```
NASA-RUL-Assignment/
├── code/
│   ├── main_rul.py                     # Main script to run the project
│   ├── data_preprocessing.py           # Data loading and preprocessing
│   ├── model_building.py               # LSTM/GRU model creation and training
│   ├── hyperparameter_tuning.py        # Hyperparameter tuning script
│   ├── evaluation.py                   # Model evaluation and comparison
│   ├── data/                           # Directory for storing datasets and model outputs
│   │   ├── train_FD001.txt             # Training data for FD001
│   │   ├── test_FD001.txt              # Testing data for FD001
│   │   ├── RUL_FD001.txt               # Ground truth RUL for FD001
│   │   ├── train_FD002.txt             # Training data for FD002
│   │   └── ...                         # Other dataset files
│   ├── models/                         # Trained LSTM/GRU models
│   │   └── FD001_the_best_lstm.keras   # Best LSTM model for FD001
│   ├── requirements.txt                # List of Python dependencies
├── report_rul.pdf                      # Final report document
├── visualizations/                     # Model performance visualizations
│   ├── FD001_accuracy_plot_lstm.png    # Accuracy plot for LSTM on FD001
│   └── ...                             # Other plots (RUL predictions, comparisons)
└── README.md                           # Project overview and instructions
```

---

## How to Run

### 1. Install Dependencies
Before running the code, ensure that all required Python libraries are installed. You can do this by running:
```bash
pip install -r code/requirements.txt
```

### 2. Run the Main Script
You can run the project through the **main_rul.py** file which provides a command-line interface (CLI). In the terminal, navigate to the project directory and run:
```bash
python code/main_rul.py
```

### 3. Command-Line Interface
Upon running the script, you'll be presented with a menu to select the dataset and whether you want to perform hyperparameter tuning:
```
== Turbofan Engine Maintenance Predictor ==
Select Which Dataset You Want to Load:
1. FD001
2. FD001 + Tuning
3. FD002
4. FD002 + Tuning
5. FD001 & FD002
6. FD001 & FD002 + Tuning
7. Exit
== Turbofan Engine Maintenance Predictor ==

Enter your choice (1/2/3/4/5/6/7):
```

### 4. Train the Model
If you choose an option with tuning, the system will run a hyperparameter search for LSTM and GRU models and display the training progress. For example:
```
Training LSTM model for dataset FD001 with units=32, dropout_rate=0.1, batch_size=16, epochs=100, optimizer=adam
Epoch 19: early stopping
Time spent training LSTM model for dataset FD001: 6.26 minutes
>> Training Accuracy: 98.06%
>> Validation Accuracy: 97.67%
>> Accuracy Gap: 0.39%
```

### 5. Check the Results
Once the training completes, you can view the results:
- **CSV Files**: Check prediction results in `data/csv/` (e.g., `data/csv/FD001_predictions_lstm.csv`).
- **Visualizations**: View performance plots in the `visualizations/` folder (e.g., accuracy and loss plots for each model).

---

## Model Development
We implement two recurrent neural network models:
- **LSTM (Long Short-Term Memory)**
- **GRU (Gated Recurrent Unit)**

Both models are evaluated on the FD001 and FD002 datasets with varying hyperparameters (units, dropout rate, etc.). Hyperparameter tuning is done using grid search.

### Key Libraries Used:
- **Keras/TensorFlow**: For model building and training
- **Pandas/Numpy**: For data loading and manipulation
- **Matplotlib**: For plotting performance metrics

---

## Model Evaluation
We use several metrics to evaluate the model performance:
- **Accuracy**: Proportion of correct predictions
- **Precision**: Proportion of true positives among predicted positives
- **Recall**: Proportion of true positives among actual positives
- **F1-Score**: Harmonic mean of precision and recall

The models are evaluated on both training and testing datasets, with comparison plots generated for LSTM and GRU models.

### Example Evaluation Output:
```
>> Accuracy: 97.80%
>> Precision (LSTM) = 98.12% 
>> Recall (LSTM) = 96.50%
>> F1-score (LSTM) = 97.30%
```

---

## Report
For detailed methodology, results, and analysis, refer to the `report_rul.pdf` in the root directory of the project.

---

## Citation
Starter Python Notebooks provided by class lecture helped to reduce the time required to complete this assignment. The provided notebook includes essential code for data loading, preprocessing, model structure, evaluation, and GitHub setup.

---

## License
This project is licensed under the MIT License.

---

## Requirements (requirements.txt)

```
pandas
numpy
matplotlib
scikit-learn
keras
tensorflow
```
