#!/usr/bin/env python
# coding: utf-8

#Import the required libraries such as pandas, numpy, matplotlib, sklearn.naive_bayes, and GaussianNB
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import MultiFunctional
import matplotlib.pyplot as plt


#Call the MultiFunctional method to read csv files, convert to numpy and spli the x and y sets
x_train_alphabet, y_train_alphabet = MultiFunctional.dataToNumpy('train_1.csv')
x_test_alphabet, y_test_alphabet = MultiFunctional.dataToNumpy('test_with_label_1.csv')

x_train_symbols, y_train_symbols = MultiFunctional.dataToNumpy('train_2.csv')
x_test_symbols, y_test_symbols = MultiFunctional.dataToNumpy('test_with_label_2.csv')


#Use the GaussianNB classifier and its method fit to train the alphabet model
#Use the predict method that takes the x_test_alphabet as an input to predcit the y_test_alphabet and assign it to results alphabet variable
model_alphabet = GaussianNB()
model_alphabet.fit(x_train_alphabet, y_train_alphabet)
predicted_alphabet = model_alphabet.predict(x_test_alphabet)



#use the score method to calculate the precentage of similarity be
#score_alphabet = model_alphabet.score(x_test_alphabet, y_test_alphabet)   # gives the percentage of similarity between two sets
#print("Alphabet similarity score is",score_alphabet*100, "%")


#Call the To_csv method to output the csv files that contains the alphabet results information such as the confusion matrix, clasification report ..
MultiFunctional.To_csv("GNB", "DS1", predicted_alphabet, y_test_alphabet)


#Use the GaussianNB classifier and its method fit to train the symbols model
#Use the predict method that takes the x_test_symbols as an input to predcit the y_test_symbols and assign it to results symbols variable
model_symbols = GaussianNB()
model_symbols.fit(x_train_symbols, y_train_symbols)
results_symbols = model_symbols.predict(x_test_symbols)


#score_symbols = model_symbols.score(x_test_symbols, y_test_symbols)   # gives the percentage of similarity between two sets
#print("Symbols similarity score is", score_symbols*100, "%")

#Call the To_csv method to output the csv files that contains the symbols results information such as the confusion matrix, clasification report ..
MultiFunctional.To_csv("GNB", "DS2", results_symbols, y_test_symbols)
