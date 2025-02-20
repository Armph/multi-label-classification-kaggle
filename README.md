# Multi-Label Classification with BERT

This project focuses on solving a multi-label classification problem across 18 subject areas of engineering using the BERT (Bidirectional Encoder Representations from Transformers) model. The goal is to classify engineering-related text data (titles and abstracts) into one or more of the 18 predefined categories.

## Table of Contents

1. [Introduction](#introduction)
2. [Data Preparation](#data-preparation)
3. [Model](#model)
4. [Results](#results)
5. [Discussion](#discussion)
6. [Conclusion](#conclusion)
7. [How to Run the Code](#how-to-run-the-code)
8. [Dependencies](#dependencies)
9. [Notes](#notes)

## Introduction

The problem involves classifying text data into multiple labels across 18 engineering subject areas. To solve this, I used **BERT**, a transformer-based model developed by Google, known for its effectiveness in natural language processing (NLP) tasks. Specifically, I used the `bert-base-uncased` model from Hugging Face due to its suitability for case-insensitive text data and limited computational resources.

## Data Preparation

1. **Data Combination**: The "Title" and "Abstract" fields were combined into a single field called "Combined" to improve model effectiveness.
2. **Text Preprocessing**: The text data was preprocessed by:
   - Lowercasing the text.
   - Removing stopwords (except "not" and "can").
   - Removing special characters, digits, and words with a length of 2 or fewer characters.
3. **Label Binarization**: The labels were binarized using `MultilabelBinarizer` to convert them into a binary matrix format.
4. **Dataset Splitting**: The dataset was split into training and validation sets with an 80-20 split.

## Model

- **Model Architecture**: I used `BertForSequenceClassification` from Hugging Face, configured for multi-label classification with 18 labels.
- **Training**: The model was trained using the following hyperparameters:
  - Batch size: 8
  - Number of epochs: 20
  - Learning rate: 7e-5
- **Evaluation Metric**: The F1 score was used to evaluate the model's performance.

## Results

After several iterations and hyperparameter tuning, the best results achieved were:

- **Public F1 Score**: 0.57007
- **Private F1 Score**: 0.66572

The results were submitted to Kaggle, and the model performed well on the test data.

## Discussion

- **Hyperparameter Tuning**: A grid search was performed to find the best hyperparameters, including learning rate, number of epochs, and batch size. The best combination was a learning rate of 7e-5 and 20 epochs.
- **Challenges**: The process was time-consuming, and further improvements could be made with more time and computational resources.
- **Key Insights**: BERT is highly effective for multi-label classification tasks, but proper data preprocessing and hyperparameter tuning are crucial for optimal performance.

## Conclusion

BERT proved to be a robust model for multi-label classification tasks. However, achieving good results required careful data preprocessing and hyperparameter tuning. With more time, further improvements could be made by exploring additional hyperparameter combinations and model architectures.

## How to Run the Code

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Armph/multi-label-classification-kaggle.git
   cd multi-label-classification-kaggle
   ```
2. **Set Up the Environment**:
   ```bash
   conda create --name bert-multi-label python=3.8
   conda activate bert-multi-label
   pip install -r requirements.txt
   ```
3. **Run the Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```
   Then, open and run notebooks/bert_main.ipynb in the Jupyter interface.
