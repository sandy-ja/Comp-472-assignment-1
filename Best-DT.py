import pandas as pd
import MultiFunctional

#
# loading the dataset
x_train , y_train = MultiFunctional.dataToNumpy('train_1.csv')

# loading the testing file
x_test , y_test = MultiFunctional.dataToNumpy('test_with_label_1.csv')

# loading the validtion data
x_val , y_val = MultiFunctional.dataToNumpy('val_1.csv')


# Best-DT

from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier


#
parameters = {
    "criterion":["gini","entropy"],
    "max_depth": [10,None],
    "min_samples_split":range(3,43,8),
    "min_impurity_decrease":[0.001,0.2,0.3,0.04], #A node will be split if this split induces a decrease of the impurity greater than or equal to this value
    "class_weight":["balanced",None]
}
#
#
clf_GS1 = GridSearchCV(DecisionTreeClassifier(),parameters, verbose = 5)


clf_GS1.fit(x_train, y_train)


# start printing the best parameters depending on Gridsearch


print('Best Criterion:', clf_GS1.best_estimator_.get_params()['criterion'])
cr = clf_GS1.best_estimator_.get_params()['criterion']

print('Best max_depth:', clf_GS1.best_estimator_.get_params()['max_depth'])


print('Best min_samples_split:', clf_GS1.best_estimator_.get_params()['min_samples_split'])


print('Best min_impurity_decrease:', clf_GS1.best_estimator_.get_params()['min_impurity_decrease'])


print('Best class_weight:', clf_GS1.best_estimator_.get_params()['class_weight'])


print('Grid search Score :', clf_GS1.best_score_)

# Testing using the results from grid but with the DT classifier
dec_tree = DecisionTreeClassifier(criterion=cr, max_depth = clf_GS1.best_estimator_.get_params()['max_depth'] ,min_samples_split = clf_GS1.best_estimator_.get_params()['min_samples_split'], min_impurity_decrease = clf_GS1.best_estimator_.get_params()['min_impurity_decrease'], class_weight = clf_GS1.best_estimator_.get_params()['class_weight'])

# dec_tree.fit(x_train, y_train)
# results = dec_tree.predict(x_test)
final_model=clf_GS1.best_estimator_
y_pred = final_model.predict(x_test)



MultiFunctional.To_csv("Best_DT","DS1",y_pred, y_test)

# Greek letters

# loading the dataset

x_trainG , y_trainG = MultiFunctional.dataToNumpy('train_2.csv')

# loading the testing file
x_testG , y_testG = MultiFunctional.dataToNumpy('test_with_label_2.csv')

# loading the validtion data
x_valG , y_valG = MultiFunctional.dataToNumpy('val_2.csv')

clf_GS2 = GridSearchCV(DecisionTreeClassifier(),parameters, verbose = 5)


clf_GS2.fit(x_trainG, y_trainG)

# start printing the best parameters depending on Gridsearch


print('Best Criterion:', clf_GS2.best_estimator_.get_params()['criterion'])
crG = clf_GS2.best_estimator_.get_params()['criterion']

print('Best max_depth:', clf_GS2.best_estimator_.get_params()['max_depth'])


print('Best min_samples_split:', clf_GS2.best_estimator_.get_params()['min_samples_split'])


print('Best min_impurity_decrease:', clf_GS2.best_estimator_.get_params()['min_impurity_decrease'])


print('Best class_weight:', clf_GS2.best_estimator_.get_params()['class_weight'])


print('Grid search Score :', clf_GS2.best_score_)


# Testing using the results from grid but with the DT classifier
dec_treeG = DecisionTreeClassifier(criterion=crG, max_depth = clf_GS2.best_estimator_.get_params()['max_depth'] ,min_samples_split = clf_GS2.best_estimator_.get_params()['min_samples_split'], min_impurity_decrease = clf_GS2.best_estimator_.get_params()['min_impurity_decrease'], class_weight = clf_GS2.best_estimator_.get_params()['class_weight'])

# dec_tree.fit(x_train, y_train)
# results = dec_tree.predict(x_test)
final_modelG=clf_GS2.best_estimator_
y_predG = final_modelG.predict(x_testG)



MultiFunctional.To_csv("Best_DT","DS2",y_predG, y_testG)
