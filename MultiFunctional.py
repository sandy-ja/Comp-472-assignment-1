import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report


# A method that takes the name of the model, the name of the dataset, the predicted values, the targeted realValues
# And print the confusion matrix, precision, accuracy, recall, f1 score, weighted average, and macro average to CSV files
def To_csv(ML_model, dataset, predictData, realValues):
    confusion = confusion_matrix(realValues, predictData)
    classification = classification_report(predictData,realValues, output_dict=True)
    df = pd.DataFrame(classification).transpose()
    predictData  = pd.DataFrame(predictData)
    predictData.index += 1 # adjusting the index to start from 1

    predictData.to_csv("CSV_Files/" + ML_model + "-" + dataset + ".csv", header= None)
    pd.DataFrame(confusion).to_csv("CSV_Files/" + ML_model + "-" + dataset + ".csv", mode='a')
    df.to_csv("CSV_Files/" + ML_model + "-" + dataset + ".csv", mode='a')


# A method that takes csv files, convert them to numpy and split each csv file to instances and their corresponding class
def dataToNumpy(CSV_file):
    file = pd.read_csv(CSV_file, header = None).to_numpy()       #reading the training file
    x = []
    y = []
    for l in file:
        x.append(l[:-1])
        y.append(l[-1])

    return x, y
