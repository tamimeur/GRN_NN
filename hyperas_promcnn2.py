from __future__ import print_function
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop
from keras.layers.convolutional import Convolution1D, MaxPooling1D

from keras.datasets import mnist
from keras.utils import np_utils

import numpy
from matplotlib import pyplot as plt

import random
import pandas
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import scipy.stats as stats
import seaborn as sea


def data():
    '''
    Data providing function:
    This function is separated from model() so that hyperopt
    won't reload data for each evaluation run.
    '''
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []

    def oneHotEncoder(seq):
        base_dict = {u'A':[1,0,0,0],u'C':[0,1,0,0],u'G':[0,0,1,0],u'T':[0,0,0,1]}
        return numpy.array([base_dict[x] for x in seq])
    def oneHotDecoder(encseq):
        dec_seq = ""
        for x in encseq:
            if (x == numpy.array([1,0,0,0])).all():
                dec_seq += u'A'
            elif (x == numpy.array([0,1,0,0])).all():
                dec_seq += u'C'
            elif (x == numpy.array([0,0,1,0])).all():
                dec_seq += u'G'
            elif (x == numpy.array([0,0,0,1])).all():
                dec_seq += u'T'
        return dec_seq

    pdata = pandas.read_excel('nbt_sharon_pp.xlsx',header=0,parse_cols="C,G")
    df = pdata[numpy.isfinite(pdata[u' expression'])]

    X_data = numpy.empty([len(df),150,4])
    indx = 0
    for seq in df[u'sequence']:
        X_data[indx] = oneHotEncoder(seq)
        indx += 1

    Y_data = numpy.array(df[[u' expression']])

    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.15, random_state=42)
    Y_train = preprocessing.StandardScaler().fit_transform(y_train)

    Y_test = preprocessing.StandardScaler().fit_transform(y_test)
    return X_train, Y_train, X_test, Y_test


def model(X_train, Y_train, X_test, Y_test):
    '''
    Model providing function:
    Create Keras model with double curly brackets dropped-in as needed.
    Return value has to be a valid python dictionary with two customary keys:
        - loss: Specify a numeric evaluation metric to be minimized
        - status: Just use STATUS_OK and see hyperopt documentation if not feasible
    The last one is optional, though recommended, namely:
        - model: specify the model just created so that we can later use it again.
    '''
    model = Sequential()
    model.add(Convolution1D(nb_filter=30,filter_length=6,input_dim=4,input_length=150,border_mode="same", activation='relu'))
    model.add(Dropout(0.1))
    model.add(Convolution1D(nb_filter=40,filter_length=6,input_dim=4,input_length=150,border_mode="same", activation='relu'))
    model.add(Flatten())
    model.add(Dense({{choice([30, 40])}}))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    
    rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
    model.compile(loss='mean_squared_logarithmic_error', optimizer=rms, metrics=['accuracy'])

    model.fit(X_train, Y_train,
              batch_size=128,
              nb_epoch=6,
              verbose=2,
              validation_data=(X_test, Y_test))
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    numpy.random.seed(1337)

    X_train, Y_train, X_test, Y_test = data()
    print("Lengths of the DATA:")
    print(len(X_train), len(Y_train), len(X_test), len(Y_test))

    best_run, best_model = optim.minimize(model=model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())

    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))