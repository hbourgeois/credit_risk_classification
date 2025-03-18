# Credit Risk Analysis Report

## Overview

The purpose of this analysis was to build and evaluate a machine learning model to predict the credit risk of loans. Specifically, the goal was to classify loans as either "healthy" (low-risk) or "high-risk" based on financial data. This analysis helps financial institutions make informed decisions about loan approvals and risk management. The dataset contained financial information about loans, including variables such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt. The target variable, `loan_status`, indicated whether a loan was healthy (`0`) or high-risk (`1`). The objective was to predict the `loan_status` based on the other financial features.

The target variable, `loan_status`, had the following distribution:
**
    y.value_counts()**

* `0` (healthy loans): Majority of the data.
* `1` (high-risk loans): Minority of the data.

This imbalance in the dataset required careful evaluation of the model's performance, particularly for the minority class (`1`).

##### Stages of the Machine Learning Process

**Data Preparation** :

* The data was loaded from [lending_data.csv](vscode-file://vscode-app/c:/Users/hbour/AppData/Local/Programs/Microsoft%20VS%20Code/resources/app/out/vs/code/electron-sandbox/workbench/workbench.html) into a Pandas DataFrame.
* The target variable (`y`) was separated from the feature variables (`X`).

**Data Splitting** :

* The dataset was split into training and testing sets using `train_test_split` with a `random_state` of 1 to ensure reproducibility.

**Model Selection and Training** :

* A logistic regression model was chosen for its simplicity and interpretability.
* The model was instantiated with the `lbfgs` solver and trained on the training data (`X_train` and `y_train`).

**Prediction** :

* The trained model was used to predict the `loan_status` for the testing set (`X_test`).

**Evaluation** :

* The model's performance was evaluated using a confusion matrix and a classification report. Metrics such as precision, recall, and F1-score were analyzed, particularly for the minority class (`1`).

#### Methods Used

* **Logistic Regression** :
* The `LogisticRegression` algorithm from `sklearn` was used to build the model. It is a linear model suitable for binary classification tasks.
* The model faced convergence warnings, indicating the need for scaling or increasing iterations for better optimization.
* **Evaluation Metrics** :
* A confusion matrix was generated to assess the model's predictions.
* A classification report provided detailed metrics, including precision, recall, and F1-score for both classes.

This analysis demonstrated the effectiveness of logistic regression in predicting healthy loans but highlighted challenges in accurately identifying high-risk loans due to class imbalance.


## Results

The results of the analysis can be classified as  **partially successful** :

**Strengths** :

* The model performed exceptionally well in predicting the majority class (`0`, healthy loans), achieving near-perfect precision, recall, and F1-scores for this class.
* The overall accuracy of the model was very high (99%), indicating strong performance on the dataset as a whole.

**Weaknesses** :

* The model struggled to predict the minority class (`1`, high-risk loans). While the recall for this class was relatively high (94%), the precision was lower (84%), indicating that the model produced a significant number of false positives for high-risk loans.
* The class imbalance in the dataset likely contributed to the model's difficulty in accurately predicting high-risk loans.

## Summary

* The model is effective for identifying healthy loans but less reliable for detecting high-risk loans. This limitation could pose challenges in real-world applications where accurately identifying high-risk loans is critical.
* To improve the results, techniques such as oversampling the minority class (e.g., SMOTE), undersampling the majority class, or using more advanced algorithms (e.g., Random Forest or Gradient Boosting) could be explored. Additionally, scaling the data or increasing the number of iterations for logistic regression might address convergence issues.
