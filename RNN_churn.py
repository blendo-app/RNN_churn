#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 12:38:42 2017

@author: eleni
"""

from random import random
import pandas as pd
import numpy as np

from keras.models import Sequential  
from keras.layers.core import Activation, Dense, Dropout
from keras.layers.recurrent import LSTM
import matplotlib.pylab as plt
from keras import metrics

def _load_data(data, n_prev = 4):  
    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):  

    ntrn = round(len(df) * (1 - test_size))

    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)

in_out_neurons = 1349 
hidden_neurons = 300

model = Sequential()
model.add(LSTM(hidden_neurons, return_sequences=False,
               input_shape=(None, in_out_neurons)))
model.add(Dense(in_out_neurons, input_dim=hidden_neurons))  
model.add(Activation("linear"))  
model.compile(loss="binary_crossentropy", optimizer="rmsprop",  metrics=[metrics.binary_accuracy])
model.summary()

model2 = Sequential()  
model2.add(LSTM(hidden_neurons, return_sequences=True, input_shape=(None, in_out_neurons)))  
model2.add(LSTM(300, return_sequences=True))  
model2.add(Dropout(0.2))  
model2.add(LSTM(300, return_sequences=False))  
model2.add(Dropout(0.2))  
model2.add(Dense(in_out_neurons, input_dim=hidden_neurons))  
model2.add(Activation("sigmoid"))  
model2.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=[metrics.binary_accuracy])  



df = pd.read_csv('PATH_TO_CSV_FILE', sep=",")

# truncate send_time to day
df['send_time']=df['send_time'].values.astype('<M8[D]')
df['actions']=df['opens']+df['clicks']
del df['opens']
del df['clicks']
df.sort_values(by='send_time')

# group by email and send_time && pivot  
df = df.groupby(['email_address','send_time'], as_index = False).sum().pivot('email_address','send_time').fillna(0).transpose()

# replace all non  zero counts with 1
df[df != 0] = 1
df = pd.DataFrame(df)

df = df.fillna(0)
df = df.transpose()

(X_train, y_train), (X_test, y_test) = train_test_split(data)  

history = model.fit(X_train, y_train, batch_size=1, epochs=25)
history2 = model2.fit(X_train, y_train, batch_size=1, epochs=20)

predicted = model.predict(X_test)
predicted
rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
    
plt.plot(history2.history['binary_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()
# "Loss"
plt.plot(history2.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
    
plt.rcParams["figure.figsize"] = (13, 9)
plt.plot(predicted[:100][:,0],"--")
plt.plot(predicted[:100][:,1],"--")
plt.plot(y_train[:100][:,0],":")
plt.plot(y_train[:100][:,1],":")
plt.legend(["Prediction 0", "Prediction 1", "Test 0", "Test 1"])   
