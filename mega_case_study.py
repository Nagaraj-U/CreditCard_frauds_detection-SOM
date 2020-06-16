# -*- coding: utf-8 -*-
"""
Created on Sat May  2 06:50:52 2020

@author: Nagaraj U
"""

#going from unsupervised learning to supervised lerning to prodict probabilities of each customer likely to cheat

#PART 1    Self Organizing Map

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)

# Training the SOM
from minisom import MiniSom
som = MiniSom(x = 10, y = 10, input_len = 15, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)

# Visualizing the results
from pylab import bone, pcolor, colorbar, plot, show
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']
for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor = colors[y[i]],
         markerfacecolor = 'None',
         markersize = 10,
         markeredgewidth = 2)
show()

# Finding the frauds
mappings = som.win_map(X)
#frauds = np.concatenate((mappings[(8,2)], mappings[(8,2)]), axis = 0)
frauds=mappings[2,6]#change coordinates accordingly
frauds = sc.inverse_transform(frauds)

#PART 2 building ANN
customers=dataset.iloc[:,1:].values#matrix of features
is_fraud=np.zeros(len(dataset))#create vector of zeros initially(dependent variable)

for i in range(len(dataset)):
    if dataset.iloc[i,0] in frauds:
        is_fraud[i]=1
        
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
customers=sc.fit_transform(customers)

from keras.models import Sequential
from keras.layers import Dense

classifier=Sequential()

classifier.add(Dense(units=2,activation='relu',kernel_initializer='uniform',input_dim=15))

classifier.add(Dense(units=1,activation='sigmoid',kernel_initializer='uniform'))

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

classifier.fit(customers,is_fraud,epochs=5,batch_size=1)

y_pred=classifier.predict(customers)#contains probabilities

#associating customer id to the probabilities by conacatinating
y_pred=np.concatenate((dataset.iloc[:,0:1].values,y_pred),axis=1)#axis=1 horozonatal concat

#sorting probabilities
y_pred=y_pred[y_pred[:,1].argsort()]#it sorts acc to first column later rearrages customer id accordingly