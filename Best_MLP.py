
#importing libraries needed
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import MultiFunctional


param_grid = {
    'activation' : ['logistic', 'tanh', 'relu', 'identity'],
    'hidden_layer_sizes' : [(50, 50), (10, 10, 10)],          
    'solver' : ['adam', 'sgd'] 
}

# loading the dataset
x_train , y_train = MultiFunctional.dataToNumpy('train_1.csv')

# loading the testing file
x_test , y_test = MultiFunctional.dataToNumpy('test_with_label_1.csv')

# loading the validtion data (we used them for validationg and changed the hyper parameters
x_val , y_val = MultiFunctional.dataToNumpy('val_1.csv')

#creating a grid search to choose the best parameters values between the "param_grid"
gridsearch = GridSearchCV(MLPClassifier(), param_grid, verbose=2) 

##training the model 
gridsearch.fit(x_train, y_train) 

#take the best parameters values and predict the target values of the test set depend on them    
final_model=gridsearch.best_estimator_
final_model_pred = final_model.predict(x_test)

##  ******************************************** These are used for demo ******************************************
##creating a MLP-classifier model and set hidden_layer_sizes to (50,50), activation to identity, and solver to adam
#clf = MLPClassifier(activation='identity', hidden_layer_sizes=(50, 50), solver='adam')
##training the model again
#clf.fit(x_train, y_train)                       
##predicting test results 
#clf_pred = clf.predict(x_test)

#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Best MLP","DS1",final_model_pred, y_test)




#**********************************************Here is the greek letters test:********************************************   

# loading the dataset
x2_train , y2_train = MultiFunctional.dataToNumpy('train_2.csv')

# loading the testing file
x2_test , y2_test = MultiFunctional.dataToNumpy('test_with_label_2.csv')

# loading the validtion data (we used them for validationg and changed the hyper parameters
x2_val , y2_val = MultiFunctional.dataToNumpy('val_2.csv')

#creating a grid search to choose the best parameters values between the "param_grid"
gridsearch2 = GridSearchCV(MLPClassifier(), param_grid, verbose=2) 

#training the model 
gridsearch2.fit(x2_train, y2_train)

#take the best parameters values and predict the target values of the test set depend on them
final_model2=gridsearch2.best_estimator_
final_model2_pred = final_model2.predict(x2_test)

#  ******************************************** These are used for demo ******************************************
#creating a MLP-classifier model and set hidden_layer_sizes to (50,50), activation to tanh, and solver to adam
#clf2 = MLPClassifier(activation='tanh', hidden_layer_sizes=(50,50), solver='adam')
#training the model again
#clf2.fit(x2_train, y2_train)
#predicting test results 
#clf2_pred = clf2.predict(x2_test)

#Printing the results to csv file using To_csv function
MultiFunctional.To_csv("Best MLP","DS2",final_model2_pred, y2_test)

