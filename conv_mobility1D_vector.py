# -*- coding: utf-8 -*-
"""
Created on Mon May 27 18:48:58 2019

@author: dguti
"""

import numpy as np
import keras 
from keras.utils import to_categorical
from keras.layers import Input, Reshape, Dense, concatenate, Activation, TimeDistributed
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import LSTM

for size in [128]:
    for i in range(1):
        
        features = np.load("features_data.npy")
        labels = np.load("labels_data.npy")
    
        Ndatos = 8000
        train = 5600
        validation = 1600
        test = 800   
        
        #features_processed =  features.flatten().reshape((8000,3601,2))   
        #features_processed = features_processed[:,1000:1300,: ]  
        features_processed = features.reshape((8000,3601,2), order='F')
        
        x_train = features_processed[0:train]
        y_train = labels[0:train]
        
        x_test = features_processed[train:train+validation]
        y_test = labels[train:train+validation]
        
        x_validate = features_processed[train+validation:]
        y_validate = labels[train+validation:]
        
        x_train /= 1000
        x_test /= 1000
        x_validate /= 1000
        
        y_train = keras.utils.to_categorical(y_train, num_classes=4)
        y_validate = keras.utils.to_categorical(y_validate, num_classes=4)
        y_test = keras.utils.to_categorical(y_test, num_classes=4)
        
        
        model = Sequential()
        file = open("resultados_1D_best.txt", 'a')
        model.add(Conv1D(filters=size, kernel_size=16, padding='valid', activation='relu', strides=1, input_shape=(3601, 2)))
        #model.add(Conv1D(filters=64, kernel_size=8, padding='valid', activation='relu', strides=1))
        model.add(Flatten())
        #model.add(Dense(128, activation='relu'))
        #model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(4, activation='softmax'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()
        
        callbacks_list = [
        keras.callbacks.ModelCheckpoint(
            filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
            monitor='val_loss', save_best_only=True),
        keras.callbacks.EarlyStopping(monitor='acc', patience=1)]
        
        history= model.fit(x_train, y_train, epochs=90, batch_size=50, verbose=2)
        validate_loss, validate_acc = model.evaluate(x_validate, y_validate)
        file.write("1D"+","+str(size)+","+str(max(history.history["acc"]))+","+str(validate_acc))
        file.write("\n")
        file.close()

#%%
import matplotlib.pyplot as plt

plt.plot(history.history["acc"])
plt.plot(history.history["loss"])
plt.xlabel("Nº epoch")
plt.ylabel("Training metric")
plt.legend(["Accuracy", "Loss"])

#%%

test_loss, test_acc = model.evaluate(x_test, y_test)

#%%

model.save('CONV1D_best.h5')

#%% Recurrent + fully connected

from keras.layers import Input, Reshape, Dense, concatenate, Activation, TimeDistributed
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import LSTM


verbose, epochs, batch_size = 1, 15, 64
n_timesteps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
model = Sequential()
model.add(LSTM(100, input_shape=(n_timesteps,n_features)))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)

#ACCURACY 0.494

#%% RNN + CONV1D

from keras.layers.convolutional import MaxPooling1D
from keras.layers import Input, Reshape, Dense, concatenate, Activation, TimeDistributed
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers import LSTM

# define model
verbose, epochs, batch_size = 1, 25, 64
n_steps, n_features, n_outputs = x_train.shape[1], x_train.shape[2], y_train.shape[1]
# reshape data into time steps of sub-sequences
n_steps, n_length = 6, 50
x_train = x_train.reshape((x_train.shape[0], n_steps, n_length, n_features))
x_test = x_test.reshape((x_test.shape[0], n_steps, n_length, n_features))

#%%

# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'), input_shape=(None,n_length,n_features)))
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu')))
#model.add(TimeDistributed(Dropout(0.5)))
model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(100))
#model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
# evaluate model
_, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=1)


#%% Convolutional 1D paralela

from keras.layers import Input, Dense, concatenate, Activation
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout
from keras.layers import Flatten

data_input = Input(shape=(2,3601))
one_branch = Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1)(data_input)
one_branch = GlobalMaxPooling1D()(one_branch)
bi_branch = Conv1D(filters=100, kernel_size=5, padding='same', activation='relu', strides=1)(data_input)
bi_branch = GlobalMaxPooling1D()(bi_branch)
tri_branch = Conv1D(filters=100, kernel_size=10, padding='same', activation='relu', strides=1)(data_input)
tri_branch = GlobalMaxPooling1D()(tri_branch)
four_branch = Conv1D(filters=100, kernel_size=20, padding='same', activation='relu', strides=1)(data_input)
four_branch = GlobalMaxPooling1D()(four_branch)
merged = concatenate([one_branch, bi_branch, tri_branch, four_branch], axis=1)

merged = Dense(256, activation='relu')(merged)
merged = Dropout(0.2)(merged)
merged = Dense(4)(merged)
output = Activation('sigmoid')(merged)
model = Model(inputs=[data_input], outputs=[output])
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
model.summary()


from keras.callbacks import ModelCheckpoint
model.fit(x_train, y_train, batch_size=32, epochs=100)

test_loss, test_acc = model.evaluate(x_test, y_test)

#%%

from keras.models import load_model

model = load_model('CONV1D_best.h5')

#%%

test_loss, test_acc = model.evaluate(x_test, y_test)


#%%

# Look at confusion matrix 
#Note, this code is taken straight from the SKLEARN website, an nice way of viewing confusion matrix.

import itertools
import matplotlib.pyplot as plt

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Prediction')
    plt.xticks([0,1,2,3],["Manhattan", "Random Waypoint", "TLW", "Gauss-Markov"])
    plt.yticks([0,1,2,3],["Manhattan", "Random Waypoint", "TLW", "Gauss-Markov"])

from collections import Counter
from sklearn.metrics import confusion_matrix

# Predict the values from the validation dataset
Y_pred = model.predict(x_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis = 1) 
# Convert validation observations to one hot vectors
Y_true = np.argmax(y_test, axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(4))

