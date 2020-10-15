import pandas as pd    #importing librabies needed
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import confusion_matrix, classification_report
import MultiFunctional


x_train , y_train = MultiFunctional.dataToNumpy('train_1.csv')

clf = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',solver='sgd')

clf.fit(x_train, y_train)


x_test , y_test = MultiFunctional.dataToNumpy('test_with_label_1.csv')


clf_pred = clf.predict(x_test)

MultiFunctional.To_csv("Base_MLP","DS1",clf_pred, y_test)

x2_train , y2_train = MultiFunctional.dataToNumpy('train_2.csv')



x2_test , y2_test = MultiFunctional.dataToNumpy('test_with_label_2.csv')

clf2 = MLPClassifier(hidden_layer_sizes=(100),activation='logistic',solver='sgd')

clf2.fit(x2_train, y2_train)

clf2_pred = clf2.predict(x2_test)

MultiFunctional.To_csv("Base_MLP","DS2",clf2_pred, y2_test)
