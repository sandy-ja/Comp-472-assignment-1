
#importing libraries needed                                                     
from sklearn.tree import DecisionTreeClassifier    
import MultiFunctional


# loading the dataset
x_train , y_train = MultiFunctional.dataToNumpy('train_1.csv')

# loading the testing file
x_test , y_test = MultiFunctional.dataToNumpy('test_with_label_1.csv')


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


