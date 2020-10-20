#Import the required libraries such as pandas, numpy, matplotlib, sklearn.naive_bayes, and GaussianNB
import pandas as pd
import numpy as np
import seaborn as sns
import MultiFunctional
import matplotlib.pyplot as plt

#Use pandas method to read csv file, convert it to numpy and assign it to info_alphabet variable
info_alphabet = pd.read_csv('info_1.csv').to_numpy()
y_info_alphabet = info_alphabet[:, -1]  #just last column

#Use pandas method to read csv file, convert it to numpy and assign it to info_symbols variable
info_symbols = pd.read_csv('info_2.csv').to_numpy()
y_info_symbols= info_symbols[:, -1]  #just last column

#Call the MultiFunctional method to read csv files, convert to numpy and spli the x and y sets
x_train_alphabet, y_train_alphabet = MultiFunctional.dataToNumpy('train_1.csv')
x_train_symbols, y_train_symbols = MultiFunctional.dataToNumpy('train_2.csv')


#Create two dictionaries, one for alphabet, and one for symobls and add just the keys


alphabet_dictionary = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 14:0, 15:0, 16:0, 17:0, 18:0, 19:0, 20:0, 21:0, 22:0, 23:0, 24:0, 25:0}
symbols_dictionary = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
counter = 0

#Iterate on the y_train_alphabet to count the number of alphabet charachters and add it to the dictionary
for y in y_train_alphabet:
    alphabet_dictionary [int(y)] += 1
    counter += 1
#Iterate on the y_train_symbols to count the number of symbols charachters and add it to the dictionary
for y in y_train_symbols:
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
