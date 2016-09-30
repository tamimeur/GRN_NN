#import grn_feedfwd as ffn
import parse_input
import bandpass_simple3 as lab
import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD
from keras.datasets import mnist
import random


class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


np.random.seed(1337)
X_train = []
Y_train = []

X_test = []
Y_test = []

############# GENERATE LANGUAGE ############

#The Grammar
Signals = ('s')
# T = ('p1','p2','p3','g0','g1','g2')
# NN_inputs = ['p1','p2','p3','g0','g1','g2']
T = ('p1','p3','g0','g1')
#NN_inputs = ['p3','g0']
NN_inputs = []
print "Neuaral N in struct = ", NN_inputs

#Productions
P = [ ('Design',           ['R']),                  \
('Design',          ['R','C']),                     \
#('Design',          ['C','C','R']),                 \
('C',               ['Prom', 'Gene']),              \
('R',               ['Prom', 'g0']),                \
('Prom',            ['p1']),                        \
#('Prom',            ['p2']),                        \
('Prom',            ['p3']),                        \
('Gene',            ['g1']),                        \
('Gene',            ['g2'])                         ]

language = parse_input.getLanguage(P,T)

print "Language = ", language
print "Num Designs = ", len(language)
# pred_design = ['p3','g2','p2','g1','p1','g0']
# language.remove(pred_design)
print "Language = ", language
print "Num Designs = ", len(language)
print "Longest design ", max(language, key=len)

############# GENERATE N.N. MODEL ############
start_time = time.time()
#num_inputs = len(T) + len(Signals)
num_inputs = len(max(language, key=len)) + len(Signals)
#num_hidden = num_inputs + (num_inputs/2)
num_hidden = 2
print 'num inputs = ', num_inputs
print 'num hidden units = ', num_hidden
print 'Compiling Model ... '
model = Sequential()
#model.add(Dense(num_hidden, input_dim=num_inputs)) #, init='zero'))
model.add(Dense(num_hidden, input_dim=num_inputs)) #, init='zero'))
model.add(Activation('sigmoid'))
model.add(Dropout(0.3))
model.add(Dense(1))
model.add(Activation('linear'))

rms = RMSprop()
#model.compile(loss='sparse_categorical_crossentropy', optimizer=rms)
sgd = SGD(lr=0.01)
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
print 'Model compield in {0} seconds'.format(time.time() - start_time)


############# RUN AGENT ############
#cur_design = ['p1','g0']
def format_data(char_data, des):
	#global NN_inputs
	global X_train
	global Y_train
	
	inducer_conc = char_data[0]
	res_conc = char_data[1]

	inputs = [0]*len(max(language, key=len))

	#parts_in_list = []
	nnindex = 0
	for part in des:
		inputs[nnindex] = (T.index(part)) + 1
		nnindex += 1

	for inducer in inducer_conc:
		new_list = list(inputs)
		new_list.append(inducer)
		X_train.append(new_list)

	for val in res_conc:
		Y_train.append([val])

	return X_train, Y_train


X_train = []
Y_train = []

cur_design = ['p3','g0']
language.remove(cur_design)
#testi = 0
#cur_char_data = lab.get_data(cur_design, testi)
#X_train, Y_train = format_data(cur_char_data, cur_design)

testi = 0
langsize = int(len(language))
while testi<langsize:
	#print "Training with design ", cur_design
	cur_char_data = lab.get_data(cur_design, testi)
	X_train, Y_train = format_data(cur_char_data, cur_design)
	cur_design = random.choice(language)
	language.remove(cur_design)
	testi += 1


spliti = 0
for data in X_train:
	if spliti%2 == 0:
		X_test.append(data)
		X_train.remove(data)
	spliti += 1

spliti = 0
for out in Y_train:
	if spliti%2 == 0:
		Y_test.append(out)
		Y_train.remove(out)
	spliti += 1
print "TRAINING DATA X: ", X_train
print "Y: ", Y_train
model.fit(X_train, Y_train, nb_epoch=5000, batch_size=5, verbose=1)


pred_goal_beh = model.predict(X_test)
print "MODEL PREDICTS FOR GOALD DES ", pred_goal_beh
print "Tru output is ", Y_test

print "Weights = ", model.get_weights()