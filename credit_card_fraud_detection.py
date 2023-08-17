import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

# 1. Data Loading and Preprocessing

def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    # Scaling the 'Amount' and 'Time' columns
    scaler = StandardScaler()
    data[['Time', 'Amount']] = scaler.fit_transform(data[['Time', 'Amount']])
    return data

# 2. Create a Balanced Dataset

def create_balanced_dataset(data):
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    print("Legit Shape:", legit.shape)
    print("Fraud Shape", fraud.shape)
    legit_sample = legit.sample(n=len(fraud))
    balanced_data = pd.concat([legit_sample, fraud], axis=0)
    return balanced_data

# 3. Splitting the data

def split_data(data):
    X = data.drop(columns='Class', axis=1)
    Y = data['Class']
    print(X)
    print(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=42)
    return X_train, X_test, Y_train, Y_test

# 4. Model Training and Evaluation

def train_and_evaluate_logistic_regression(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    # Predictions
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    # Evaluations
    train_accuracy = accuracy_score(Y_train, Y_pred_train)
    test_accuracy = accuracy_score(Y_test, Y_pred_test)
    classification_rep = classification_report(Y_test, Y_pred_test)
    conf_matrix = confusion_matrix(Y_test, Y_pred_test)
    roc_auc = roc_auc_score(Y_test, Y_pred_test)
    return train_accuracy, test_accuracy, classification_rep, conf_matrix, roc_auc

# Main execution

data = load_and_preprocess_data('creditcard.csv')
X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = split_data(data)
balanced_data = create_balanced_dataset(data)
X_train_balanced, X_test_balanced, Y_train_balanced, Y_test_balanced = split_data(balanced_data)

# Logistic Regression
lr_train_accuracy, lr_test_accuracy, lr_classification_rep, lr_conf_matrix, lr_roc_auc = train_and_evaluate_logistic_regression(X_train_balanced, Y_train_balanced, X_test_balanced, Y_test_balanced)

# Evaluations for Logistic Regression
print("=== Logistic Regression Model ===")
print("\nTraining Accuracy:", lr_train_accuracy)
print("\nTest Accuracy:", lr_test_accuracy)
print("\nClassification Report:\n", lr_classification_rep)
print("\nConfusion Matrix:\n", lr_conf_matrix)
print("\nROC AUC Score:", lr_roc_auc)