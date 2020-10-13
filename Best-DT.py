#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


#get_ipython().run_line_magic('matplotlib', 'inline')




# loading the training dataset
letters = pd.read_csv('train_1.csv')



# getting the class column
y_train = letters.iloc[:,-1:]



x_train = letters.iloc[:,:-1]



# loading the testing file
test = pd.read_csv('test_with_label_1.csv')




x_test = test.iloc[:,:-1]


y_test = test.iloc[:,-1:]


# Best-DT


from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


# Creating a DecisionTreeClassifier

dec_tree = tree.DecisionTreeClassifier()



parameters = {
    "criterion":["gini","entropy"],
    "max_depth": [10,None],
    "min_samples_split":range(4,40,10),
    "min_impurity_decrease":[0.01,0.2,0.3,0.04],
    "class_weight":["balanced",None]
}


# In[177]:


clf_GS = GridSearchCV(dec_tree,parameters)


# In[178]:


clf_GS.fit(x_train, y_train)




# start printing the best parameters depending on Gridsearch


print('Best Criterion:', clf_GS.best_estimator_.get_params()['criterion'])
cr = clf_GS.best_estimator_.get_params()['criterion']

print('Best max_depth:', clf_GS.best_estimator_.get_params()['max_depth'])


print('Best min_samples_split:', clf_GS.best_estimator_.get_params()['min_samples_split'])


print('Best min_impurity_decrease:', clf_GS.best_estimator_.get_params()['min_impurity_decrease'])


print('Best class_weight:', clf_GS.best_estimator_.get_params()['class_weight'])


# In[168]:


# printing the tree
#print(clf_GS.best_estimator_.get_params()[dec_tree])


# Testing prediction using the grid

val = pd.read_csv('val_1.csv')

x_val = val.iloc[:,:-1]

y_val = val.iloc[:,-1:]
#results = clf_GS.predict(x_val)


# Testing using the results from grid but with the DT classifier


dec_tree = DecisionTreeClassifier(criterion=cr, max_depth = clf_GS.best_estimator_.get_params()['max_depth'] ,min_samples_split = clf_GS.best_estimator_.get_params()['min_samples_split'], min_impurity_decrease = clf_GS.best_estimator_.get_params()['min_impurity_decrease'], class_weight = clf_GS.best_estimator_.get_params()['class_weight'])



dec_tree.fit(x_train, y_train)
#dec_tree.fit(x_val, y_val)


results = dec_tree.predict(x_test)
#results = dec_tree.predict(x_val)







#clf_GS.score(x_val,y_val)
clf_GS.score(x_test,y_test)



print(classification_report(y_test,results))
print(confusion_matrix(y_test,results))


# clf_GS.best_estimator_
# clf_GS.best_score_
# clf_GS.best_params_
# tree.plot_tree(clf_GS)
