# Credit Card Fraud Detection

This project focuses on detecting credit card fraud using a classification model trained on an unbalanced dataset. The dataset is sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Overview

In this exercise, we initially trained our model using unbalanced data to evaluate the accuracy of the model. The performance metrics, including precision, recall, and the confusion matrix, were calculated for both validation and test datasets.

## Dataset

The dataset used in this project contains credit card transactions, with a highly imbalanced class distribution where fraudulent transactions are significantly less common than non-fraudulent ones. This poses a challenge for the model, as it needs to correctly identify the rare fraudulent cases.

## Methodology

1. **Data Preprocessing**: 
   - Handled the unbalanced nature of the dataset.
   - Performed necessary data cleaning and transformation.

2. **Model Training**:
   - The model was trained using the unbalanced dataset.
   - Various machine learning algorithms were explored and compared to determine the most effective approach.

3. **Evaluation**:
   - The model's performance was evaluated using precision, recall, and the confusion matrix.
   - Evaluations were done on both the validation dataset and the test dataset to assess the generalizability of the model.

## Results

After training the model with unbalanced data, the following performance metrics were used to accessed the model:

- **Precision**
- **Recall** 
- **Confusion Matrix**

These metrics provide insight into the effectiveness of the model in detecting fraudulent transactions.

## Installation and Usage

This project was developed in Scala. To run the project, ensure that you have a working Scala environment set up. You can follow these general steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your_username/credit-card-fraud-detection.git](https://github.com/Franosei/Credit-card-fraud-detection.git
    cd credit-card-fraud-detection
    ```

2. **Run the Scala application**:
    - Depending on your setup, you can use an IDE like IntelliJ IDEA or a build tool like SBT to compile and run the application.
    - Make sure all necessary dependencies are included in your `build.sbt` or equivalent file.


