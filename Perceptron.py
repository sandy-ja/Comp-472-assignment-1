#importing libraries needed
from sklearn.linear_model import Perceptron
import MultiFunctional 



# loading the dataset
x_train , y_train = MultiFunctional.dataToNumpy('train_1.csv')

# loading the testing file
x_test , y_test = MultiFunctional.dataToNumpy('test_with_label_1.csv')


#creating a perceptron model
per = Perceptron()

#training the model 
per.fit(x_train, y_train)

#predicting test results 
per_pred = per.predict(x_test)


#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Perceptron","DS1",per_pred, y_test)



#**********************************************Here is the greek letters test:********************************************   

# loading the dataset
x2_train , y2_train = MultiFunctional.dataToNumpy('train_2.csv')

# loading the testing file
x2_test , y2_test = MultiFunctional.dataToNumpy('test_with_label_2.csv')


#creating a perceptron model
per2 = Perceptron()

#training the model 
per2.fit(x2_train, y2_train)

#predicting test results 
per2_pred = per2.predict(x2_test)


#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Perceptron","DS2",per2_pred, y2_test)


