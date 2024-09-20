
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
For the project submission, we didn't include the data/csv and data/model folders in this repository. Please add them manually after downloading or cloning the repo.

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
│   ├── model/                         # Trained LSTM/GRU models
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
2. FD001 << Tuning
3. FD002
4. FD002 << Tuning
5. FD001 & FD002
6. FD001 & FD002 << Tuning
7. FD001 << Evaluate (Pre-trained)
8. FD002 << Evaluate (Pre-trained)
9. FD001 & FD002 << Evaluate (Pre-trained)
10. Exit
== Turbofan Engine Maintenance Predictor ==

Enter your choice (1-10):
```
Notes:
- Option 1: FD001 with static parameters
- Option 2: FD001 with hyperparameter tuning
- Option 3: FD002 with static parameters
- Option 4: FD002 with hyperparameter tuning
- Option 5: Both datasets with static parameters
- Option 6: Both datasets with hyperparameter tuning
- Option 7: Evaluate the pre-trained Best Model for FD001
- Option 8: Evaluate the pre-trained Best Model for FD002
- Option 9: Evaluate the pre-trained Best Models for Both datasets
- Option 10: Exit the program

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
- **Modle Files**: The Best Model for LSTM and GRU in `data/model/` (e.g., `data/model/FD001_the_best_lstm.keras.csv`).
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

Dataset Model_Type  Units  Dropout_Rate  Batch_Size Optimizer  Train_Accuracy  Val_Accuracy  Accuracy_Gap  Train_Time
67    FD001        GRU     50           0.3          32      adam        0.983525      0.984010      0.000485    2.586073
83    FD001        GRU     64           0.1          64   rmsprop        0.982646      0.984010      0.001365    3.213751
68    FD001        GRU     64           0.3          32      adam        0.984805      0.983371      0.001434    2.978215
44    FD001        GRU     64           0.1          32      adam        0.984725      0.983051      0.001674    7.248248
95    FD001        GRU     64           0.2          64   rmsprop        0.981126      0.982731      0.001605    3.091439
..      ...        ...    ...           ...         ...       ...             ...           ...           ...         ...
0     FD001       LSTM     32           0.1          16      adam        0.984965      0.966741      0.018224    4.939663
100   FD001       LSTM     50           0.3          64   rmsprop        0.982885      0.965782      0.017104    2.091014
29    FD001       LSTM     64           0.3          16   rmsprop        0.981526      0.961625      0.019901    3.549675
41    FD001       LSTM     64           0.1          32   rmsprop        0.982086      0.959066      0.023019    2.949606
17    FD001       LSTM     64           0.2          16   rmsprop        0.982566      0.958427      0.024139    5.244556

[108 rows x 10 columns]

Best LSTM model for FD001 saved with validation accuracy: 98.24% and accuracy gap: 0.10%
Best GRU model for FD001 saved with validation accuracy: 98.40% and accuracy gap: 0.05%

Hyperparameter tuning complete for FD001. Results saved to 'data/csv/FD001_hyperparameter_results.csv'.

Model Performance on Training Data for The Best LSTM Model in FD001
>> Accuracy : 98.80%   
>> Precision (LSTM) = 98.08% 
>> Recall (LSTM) = 95.81%
>> F1-score (LSTM) = 96.93%
>> Confusion matrix:
[[12473    58]
 [  130  2970]]

Evaluating models for FD001 dataset...

Model Performance on Test Data for The Best LSTM Model in FD001
>> Accuracy: 95.70%
>> Precision (LSTM) = 100.00%
>> Recall (LSTM) = 84.00%
>> F1-score (LSTM) = 91.30%
>> Confusion matrix:
[[68  0]
 [ 4 21]]

Accuracy  Precision  Recall  F1-score
LSTM                 0.956989   1.000000    0.84  0.913043
GRU                  0.946237   0.954545    0.84  0.893617
Template Best Model  0.940000   0.952381    0.80  0.869565
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
