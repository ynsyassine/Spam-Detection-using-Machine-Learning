#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 23:24:08 2023

@author: yassine
"""



import nltk 
nltk.download("stopwords")
nltk.download('wordnet')
import re  
import pandas  as pd  

messages = pd.read_csv("https://raw.githubusercontent.com/krishnaik06/SpamClassifier/master/smsspamcollection/SMSSpamCollection",
                        sep="\t",
                        header=None,
                        names=["label", "text"])

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# Separate the data into text and labels
y = pd.get_dummies(messages['label'])
y = y.iloc[:, 1].values

# Instantiate the stemmer and lemmatizer
ps = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Split the data into train and test sets
from sklearn.model_selection import train_test_split
X_train_raw, X_test_raw, y_train, y_test = train_test_split(messages["text"], y, test_size=0.2, random_state=42)

corpus_stem_train = []
corpus_stem_test = []
corpus_lemm_train = []
corpus_lemm_test = []

# Process train data
for text in X_train_raw:
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review_stem = [ps.stem(word_stem) for word_stem in review if word_stem not in set(stopwords.words("english"))]
    review_stem = ' '.join(review_stem)
    review_lemm = [lemmatizer.lemmatize(word_lemm) for word_lemm in review if word_lemm not in set(stopwords.words("english"))]
    review_lemm = ' '.join(review_lemm)
    corpus_stem_train.append(review_stem)
    corpus_lemm_train.append(review_lemm)

# Process test data
for text in X_test_raw:
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower()
    review = review.split()
    review_stem = [ps.stem(word_stem) for word_stem in review if word_stem not in set(stopwords.words("english"))]
    review_stem = ' '.join(review_stem)
    review_lemm = [lemmatizer.lemmatize(word_lemm) for word_lemm in review if word_lemm not in set(stopwords.words("english"))]
    review_lemm = ' '.join(review_lemm)
    corpus_stem_test.append(review_stem)
    corpus_lemm_test.append(review_lemm)

# Create the vectorizers
cv = CountVectorizer(max_features=2500)
tv = TfidfVectorizer(max_features=2500)

# Vectorize the data
X_stem_cv_train = cv.fit_transform(corpus_stem_train).toarray()
X_stem_cv_test = cv.transform(corpus_stem_test).toarray()
X_lemm_cv_train = cv.fit_transform(corpus_lemm_train).toarray()
X_lemm_cv_test = cv.transform(corpus_lemm_test).toarray()

X_stem_tv_train = tv.fit_transform(corpus_stem_train).toarray()
X_stem_tv_test = tv.transform(corpus_stem_test).toarray()
X_lemm_tv_train = tv.fit_transform(corpus_lemm_train).toarray()
X_lemm_tv_test = tv.transform(corpus_lemm_test).toarray()

# Train the models
from sklearn.naive_bayes import MultinomialNB

MLYNB_STEM_CV = MultinomialNB().fit(X_stem_cv_train, y_train)
MLYNB_STEM_TV = MultinomialNB().fit(X_stem_tv_train, y_train)

MLYNB_LEMM_CV = MultinomialNB().fit(X_lemm_cv_train, y_train)
MLYNB_LEMM_TV = MultinomialNB().fit(X_lemm_tv_train, y_train)

# Evaluate the models
from sklearn.metrics import classification_report, accuracy_score

y_pred_stem_cv = MLYNB_STEM_CV.predict(X_stem_cv_test)
y_pred_stem_tv = MLYNB_STEM_TV.predict(X_stem_tv_test)
y_pred_lemm_cv = MLYNB_LEMM_CV.predict(X_lemm_cv_test)
y_pred_lemm_tv = MLYNB_LEMM_TV.predict(X_lemm_tv_test)

# Calculate performance metrics
print("Stemmed, CountVectorizer:")
print("Accuracy:", accuracy_score(y_test, y_pred_stem_cv))
print("Classification report:\n", classification_report(y_test, y_pred_stem_cv))

print("Stemmed, TfidfVectorizer:")
print("Accuracy:", accuracy_score(y_test, y_pred_stem_tv))
print("Classification report:\n", classification_report(y_test, y_pred_stem_tv))

print("Lemmatized, CountVectorizer:")
print("Accuracy:", accuracy_score(y_test, y_pred_lemm_cv))
print("Classification report:\n", classification_report(y_test, y_pred_lemm_cv))

print("Lemmatized, TfidfVectorizer:")
print("Accuracy:", accuracy_score(y_test, y_pred_lemm_tv))
print("Classification report:\n", classification_report(y_test, y_pred_lemm_tv))

import matplotlib.pyplot as plt

# Calculate accuracy scores
accuracy_stem_cv = accuracy_score(y_test, y_pred_stem_cv)
accuracy_stem_tv = accuracy_score(y_test, y_pred_stem_tv)
accuracy_lemm_cv = accuracy_score(y_test, y_pred_lemm_cv)
accuracy_lemm_tv = accuracy_score(y_test, y_pred_lemm_tv)

# Prepare data for bar plot
labels = ['Stem, CountVectorizer', 'Stem, TfidfVectorizer', 'Lemm, CountVectorizer', 'Lemm, TfidfVectorizer']
accuracy_values = [accuracy_stem_cv, accuracy_stem_tv, accuracy_lemm_cv, accuracy_lemm_tv]

# Create a bar plot
plt.figure(figsize=(10, 5))
plt.bar(labels, accuracy_values)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison of Different Preprocessing and Vectorization Methods')
plt.ylim([min(accuracy_values)-0.01, max(accuracy_values)+0.01])

# Show the plot
plt.show()

