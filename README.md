# Spam Detection using Machine Learning

This project aims to develop an effective spam detection system using machine learning techniques. The system is designed to classify text messages as either spam or ham (non-spam), helping users filter out unwanted messages and maintain a clutter-free inbox.

## Overview

In this project, we explore various preprocessing techniques, feature extraction methods, and machine learning algorithms to build a robust spam detection model. The main components of the project include:

### Data Preprocessing:

- Tokenization
- Stopword removal
- Stemming and Lemmatization

### Feature Extraction:

- CountVectorizer
- TfidfVectorizer

### Model Training and Evaluation:

- Naive Bayes Classifier
- Model performance comparison

## Dataset

The dataset used in this project is the SMS Spam Collection dataset, which contains a total of 5,572 labeled text messages. The dataset is publicly available and can be accessed [here](https://raw.githubusercontent.com/krishnaik06/SpamClassifier/master/smsspamcollection/SMSSpamCollection).

## Dependencies

To run this project, you will need the following Python libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- nltk

## Getting Started

1. Clone the repository to your local machine.
2. Install the required Python libraries using `pip install -r requirements.txt`.
3. Run the Python script `spam_detection.py` to train and evaluate the spam detection models.

## Results

The performance of the different models is compared using accuracy scores and classification reports. The results are visualized in a bar plot, showcasing the effectiveness of each model.




