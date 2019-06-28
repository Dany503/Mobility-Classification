# -*- coding: utf-8 -*-
"""
Created on Mon May 27 10:19:20 2019

@author: dguti
"""

import numpy as np

features = np.load("features_data.npy")
labels = np.load("labels_data.npy")

Ndatos = 8000
train = 5600
validation = 1600
test = 800   

x_train = features[0:train]
y_train = labels[0:train]

x_test = features[train:train+validation]
y_test = labels[train:train+validation]

x_validate = features[train+validation:]
y_validate = labels[train+validation:]


x_train /= 1000
x_test /= 1000
x_validate /= 1000


print(x_train.shape)
print(x_test.shape)

#%%
import keras 
from keras.utils import to_categorical

y_train = keras.utils.to_categorical(y_train, num_classes=4)
y_validate = keras.utils.to_categorical(y_validate, num_classes=4)
y_test = keras.utils.to_categorical(y_test, num_classes=4)


#%%
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import sgd

#neurons = [16, 32, 64, 128, 256]
#neurons = [512]
batch_size = 50
num_classes = 4
epochs=90

#for nodes_1, nodes_2 in zip(neurons_1, neurons_2):
#file = open("resultados_4layer.txt", 'a')        
model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(7202,)))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy',
      optimizer='adam',
      metrics=['accuracy'])

history = model.fit(x_train, y_train,
  batch_size=batch_size,
  epochs=epochs,
  verbose=1
  )

validate_loss, validate_acc = model.evaluate(x_validate, y_validate)

#print('Test loss:', test_loss)
#print('Test accuracy:', test_acc)

#file.write(str(i)+","+"MLP"+","+str(256)+","+str(128)+","+str(64)+","+str(max(history.history["acc"]))+","+str(validate_acc))
#file.write("\n")
#file.close()

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

model.save('MLP_best.h5')

#%%

from keras.models import load_model

model = load_model('MLP_best.h5')

#%%

import matplotlib.pyplot as plt

x_rwp = np.array([x_test[1]])
x_tlw = np.array([x_test[2]])
y_predicted = model.predict_classes(x_rwp)
print(y_predicted)

#%%

x_data_rwp = x_rwp[0,0:3601] * 1000
y_data_rwp = x_rwp[0,3601:] * 1000

x_data_tlw = x_tlw[0,0:3601] * 1000
y_data_tlw = x_tlw[0,3601:] * 1000

plt.figure(figsize=(10,6))

plt.subplot(1,2,1)
plt.scatter(x_data_rwp, y_data_rwp)
plt.xlim([0,1000])
plt.ylim([0,1000])
plt.xlabel("x")
plt.ylabel("y")
plt.title("Waypoint predicted as TLW")

plt.subplot(1,2,2)
plt.scatter(x_data_tlw, y_data_tlw)
plt.xlim([0,1000])
plt.ylim([0,1000])
plt.xlabel("x")
plt.ylabel("y")
plt.title("TLW predicted as Waypoint")


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

