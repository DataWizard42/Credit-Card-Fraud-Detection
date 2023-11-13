# Credit Card Fraud Detection with TensorFlow
This repository contains a credit card fraud detection model implemented in TensorFlow, aiming to identify fraudulent transactions in credit card data.

# Dataset

The dataset used for this project is sourced from Kaggle, containing credit card transactions with labeled fraud and non-fraud cases.

# Exploratory Data Analysis (EDA)
The EDA section includes visualizations of the class distribution and transaction amounts for fraud and normal transactions. The goal is to gain insights into the characteristics of the data.

# Model Architecture
The fraud detection model is implemented using a neural network with the following architecture:

Input Layer: Dense layer with ReLU activation and L2 regularization
Dropout Layer: 20% dropout rate
Hidden Layer: Dense layer with ReLU activation and L2 regularization
Dropout Layer: 20% dropout rate
Output Layer: Dense layer with sigmoid activation
Data Preprocessing
The dataset is split into training, validation, and test sets. Standard scaling is applied to normalize the data. Additionally, class weights are calculated to handle the imbalanced nature of the dataset.

# Model Training
The model is compiled with binary cross-entropy loss and custom metrics, including precision, recall, false positives, false negatives, true positives, and true negatives. The training process is configured with early stopping.

# Usage
Install the required dependencies:

# bash  
# Copy code  
pip install -r requirements.txt
Run the credit_card_fraud_detection.ipynb notebook to train and evaluate the model.

# Results
The model's performance metrics on the validation set are tracked during training. Early stopping is implemented to prevent overfitting.

Future Improvements
Fine-tuning hyperparameters for better model performance.
Exploring other architectures and algorithms.
Enhancing the model's interpretability and explainability.
Feel free to explore the code, experiment with the parameters, and contribute to further improvements!
