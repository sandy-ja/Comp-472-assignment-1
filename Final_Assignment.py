import pandas as pd
import numpy as np
import seaborn as sns
import MultiFunctional
import matplotlib.pyplot as plt


#Call the MultiFunctional method to read csv files, convert to numpy and split the x and y sets
x_train, y_train = MultiFunctional.dataToNumpy('train_1.csv')
x2_train, y2_train = MultiFunctional.dataToNumpy('train_2.csv')

# loading the testing file
x_test , y_test = MultiFunctional.dataToNumpy('test_with_label_1.csv')
x2_test , y2_test = MultiFunctional.dataToNumpy('test_with_label_2.csv')


#################################### Part 1 #########################################

#Use pandas method to read csv file, convert it to numpy and assign it to info_alphabet variable
info_alphabet = pd.read_csv('info_1.csv').to_numpy()
y_info_alphabet = info_alphabet[:, -1]  #just last column

#Use pandas method to read csv file, convert it to numpy and assign it to info_symbols variable
info_symbols = pd.read_csv('info_2.csv').to_numpy()
y_info_symbols= info_symbols[:, -1]  #just last column

#Call the MultiFunctional method to read csv files, convert to numpy and split the x and y sets
x_train, y_train = MultiFunctional.dataToNumpy('train_1.csv')
x2_train, y2_train = MultiFunctional.dataToNumpy('train_2.csv')


#Create two dictionaries, one for alphabet, and one for symobls and add just the keys


alphabet_dictionary = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0}
symbols_dictionary = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
counter = 0

#Iterate on the y_train to count the number of alphabet charachters and add it to the dictionary
for y in y_train:
    alphabet_dictionary [int(y)] += 1
    counter += 1
#Iterate on the y2_train to count the number of symbols charachters and add it to the dictionary
for y in y2_train:
    symbols_dictionary[int(y)] += 1
    counter += 1


alphabet_dictionary.values()
ad = alphabet_dictionary .values()

symbols_dictionary.values()
sd = symbols_dictionary.values()

#use the matplolib library to graph the required alphabet and symbols

fig, axs = plt.subplots(2)
fig.suptitle('The Distribution of Numbers of Alphabet and Symbol letters ')
axs[0].bar(y_info_alphabet, ad, color= "purple")
axs[1].bar(y_info_symbols, sd, color="pink")
plt.xticks(fontsize=12, rotation=90)
plt.savefig("alphabet and symbols graph.png")
plt.show()

########################### Part2 #######################################

########################### GNB #######################################

from sklearn.naive_bayes import GaussianNB


#Using the GaussianNB classifier and its method fit to train the alphabet model
#Using the predict method that takes the x_train as an input to predict the y_train and assign it to results alphabet variable
model_alphabet = GaussianNB()
model_alphabet.fit(x_train, y_train)
predicted_alphabet = model_alphabet.predict(x_test)


#Call the To_csv method to output the csv files that contains the alphabet results information such as the confusion matrix, clasification report ..
MultiFunctional.To_csv("GNB", "DS1", predicted_alphabet, y_test)


#Use the GaussianNB classifier and its method fit to train the symbols model
#Use the predict method that takes the x2_train as an input to predcit the y2_train and assign it to results symbols variable
model_symbols = GaussianNB()
model_symbols.fit(x2_train, y2_train)
results_symbols = model_symbols.predict(x2_test)


#Call the To_csv method to output the csv files that contains the symbols results information such as the confusion matrix, clasification report ..
MultiFunctional.To_csv("GNB", "DS2", results_symbols, y2_test)


########################### Base-DT #######################################


from sklearn.tree import DecisionTreeClassifier

#Creating a decision-tree-classifier model and set the criterion parameter to entropy
clf = DecisionTreeClassifier(criterion="entropy")

#training the model
clf = clf.fit(x_train,y_train)

#predicting test results
pred_clf = clf.predict(x_test)


#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Base-DT","DS1",pred_clf, y_test)



#**********************************************Here is the greek letters test:********************************************

# loading the dataset
x2_train , y2_train = MultiFunctional.dataToNumpy('train_2.csv')

# loading the testing file
x2_test , y2_test = MultiFunctional.dataToNumpy('test_with_label_2.csv')


#creating another decision-tree-classifier and set criterion to entropy
clf2 = DecisionTreeClassifier(criterion="entropy")

#training the model
clf2 = clf2.fit(x2_train,y2_train)

#predicting test results
pred_clf2 = clf2.predict(x2_test)


#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Base-DT","DS2",pred_clf2, y2_test)

########################### Best-DT #######################################

from sklearn.model_selection import GridSearchCV

################ Latin Letters ###############
# Creating the tree object
dec_tree = DecisionTreeClassifier(criterion="entropy", max_depth = None ,min_samples_split = 3 , min_impurity_decrease = 0.001 , class_weight = None)

#Training the model
dec_tree.fit(x_train, y_train)

#Finding the prediected output for the new dataset
y_pred = dec_tree.predict(x_test)

#Saving the output in a CSV file
MultiFunctional.To_csv("Best_DT","DS1",y_pred, y_test)


################ Greek Letters ###############

# Creating the tree object
dec_treeG = DecisionTreeClassifier(criterion="gini", max_depth = None ,min_samples_split = 3 , min_impurity_decrease = 0.001 , class_weight = None)

#Training the model
dec_treeG.fit(x2_train, y2_train)

#Finding the prediected output for the new dataset
y2_pred = dec_treeG.predict(x2_test)

#Saving the output in a CSV file
MultiFunctional.To_csv("Best_DT","DS2",y2_pred, y2_test)

########################### PER #######################################

#importing libraries needed
from sklearn.linear_model import Perceptron

#creating a perceptron model
per = Perceptron()

#training the model
per.fit(x_train, y_train)

#predicting test results
per_pred = per.predict(x_test)


#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Perceptron","DS1",per_pred, y_test)



#**********************************************Here is the greek letters test:********************************************

#creating a perceptron model
per2 = Perceptron()

#training the model
per2.fit(x2_train, y2_train)

#predicting test results
per2_pred = per2.predict(x2_test)


#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Perceptron","DS2",per2_pred, y2_test)

########################### Base-MLP #######################################

from sklearn.neural_network import MLPClassifier

#creating a MLP-classifier model and set hidden_layer_sizes to 100,activation to logistic, and solver to sgd
clf = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',solver='sgd')

#training the model
clf.fit(x_train, y_train)

#predicting test results
clf_pred = clf.predict(x_test)



#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Base-MLP","DS1",clf_pred, y_test)



#**********************************************Here is the greek letters test:********************************************


#creating a MLP-classifier model and set hidden_layer_sizes to 100,activation to logistic, and solver to sgd
clf2 = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',solver='sgd')

#training the model
clf2.fit(x2_train, y2_train)

#predicting test results
clf2_pred = clf2.predict(x2_test)


#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Base-MLP","DS2",clf2_pred, y2_test)

########################### Best-MLP #######################################

##creating a MLP-classifier model and set hidden_layer_sizes to (50,50), activation to identity, and solver to adam
clf = MLPClassifier(activation='identity', hidden_layer_sizes=(50, 50), solver='adam')
#training the model again
clf.fit(x_train, y_train)
#predicting test results
clf_pred = clf.predict(x_test)

#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Best MLP","DS1",clf_pred, y_test)

#creating a MLP-classifier model and set hidden_layer_sizes to (50,50), activation to tanh, and solver to adam
clf2 = MLPClassifier(activation='tanh', hidden_layer_sizes=(50,50), solver='adam')
# training the model again
clf2.fit(x2_train, y2_train)
# predicting test results
clf2_pred = clf2.predict(x2_test)

#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Best MLP","DS2",clf2_pred, y2_test)
