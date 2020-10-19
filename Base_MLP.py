#importing libraries needed
from sklearn.neural_network import MLPClassifier
import MultiFunctional


# loading the dataset
x_train , y_train = MultiFunctional.dataToNumpy('train_1.csv')

# loading the testing file
x_test , y_test = MultiFunctional.dataToNumpy('test_with_label_1.csv')


#creating a MLP-classifier model and set hidden_layer_sizes to 100,activation to logistic, and solver to sgd
clf = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',solver='sgd')

#training the model
clf.fit(x_train, y_train)

#predicting test results 
clf_pred = clf.predict(x_test)                           



#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Base-MLP","DS1",clf_pred, y_test)



#**********************************************Here is the greek letters test:********************************************   

# loading the dataset
x2_train , y2_train = MultiFunctional.dataToNumpy('train_2.csv')

# loading the testing file
x2_test , y2_test = MultiFunctional.dataToNumpy('test_with_label_2.csv')


#creating a MLP-classifier model and set hidden_layer_sizes to 100,activation to logistic, and solver to sgd
clf2 = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',solver='sgd')

#training the model
clf2.fit(x2_train, y2_train)

#predicting test results 
clf2_pred = clf2.predict(x2_test)


#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Base-MLP","DS2",clf2_pred, y2_test)


