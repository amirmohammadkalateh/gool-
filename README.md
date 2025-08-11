Deep Learning Model for Air Quality Index (AQI) Prediction
Overview
This project implements a deep learning model using TensorFlow and Keras to predict the Air Quality Index (AQI) based on a given dataset. The script performs a manual grid search to find the optimal hyperparameters and incorporates a range of advanced techniques to prevent common issues like overfitting and vanishing gradients.

Features
Deep Learning Model: A sequential neural network built with TensorFlow and Keras.

Data Preprocessing: Loads and prepares the final_dataset.csv file, including feature standardization with StandardScaler.

Manual Grid Search: A custom implementation to efficiently search for the best combination of hyperparameters, replacing the problematic GridSearchCV wrapper.

Overfitting Prevention:

L1 and L2 Regularization

Max-norm Regularization

Dropout Layers

EarlyStopping Callback

Vanishing Gradient Prevention:

He Weight Initialization (he_uniform)

Batch Normalization Layers

Non-saturating Activation Function (relu)

Gradient Clipping

Model Evaluation: The final, best-performing model is evaluated on a dedicated test set to measure its performance.

Prerequisites
To run this code, you need to have the following Python libraries installed. You can install them using pip:

pip install pandas numpy tensorflow scikit-learn

Note: You may need to install scikeras if you choose to experiment with that library, but the current version of the code uses a manual grid search to avoid known compatibility issues.

Usage
Save the Dataset: Ensure your dataset is saved as final_dataset.csv in the same directory as the Python script.

Run the Script: Execute the Python script from your terminal.

python your_script_name.py

The script will handle all the steps, from data loading to hyperparameter tuning and final evaluation.

Code Walkthrough
The script is divided into several logical steps:

Step 1: Imports: All necessary libraries are imported. A random seed is set for reproducibility.

Step 2: Data Preparation: The final_dataset.csv file is loaded. The Date column is dropped, and the features (X) and target (y) are defined. The data is then split into training and testing sets and standardized.

Step 3: Model Creation Function: The create_model function defines the neural network architecture. It takes hyperparameters as arguments, allowing it to be configured for the grid search.

Step 4: Manual Grid Search: This is the core of the tuning process. It defines a dictionary of hyperparameters and iterates through all possible combinations. For each combination, it creates and trains a model, then evaluates it on a validation set to find the best-performing one.

Step 5: Results and Evaluation: After the grid search is complete, the best hyperparameters and the best model's performance on the test set are printed to the console. Example predictions are also shown.
