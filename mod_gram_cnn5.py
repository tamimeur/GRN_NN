#import grn_feedfwd as ffn
import parse_input
import bandpass_simple4 as lab
import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, SGD
from keras.datasets import mnist
from keras.layers import Merge
from keras.regularizers import l1, l2, activity_l2
import random
import pandas as pd


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

def format_data(char_data, des):
	#global NN_inputs
	global X_train
	global Y_train
	
	inducer_conc = char_data[0]
	res_conc = char_data[1]

	inputs = [0]*(len(des)-1)

	#parts_in_list = []
	nnindex = 0
	for part in des:
		if part != 'g0':
			inputs[nnindex] = (T.index(part)) + 1
			nnindex += 1

	for inducer in inducer_conc:
		new_list = list(inputs)
		new_list.append(inducer)
		X_train.append(new_list)

	for val in res_conc:
		Y_train.append([val])

	return X_train, Y_train
############# LOAD DATA - NBT PAPER ########
pdata = pd.read_excel('nbt_sharon_pp.xlsx',header=0,parse_cols="C,G")


print "Is YO DATA BOI!!! ", pdata
print "length of data ", pdata.size


############# GENERATE LANGUAGE ############

#The Grammar
Signals = ('s')
T = ('p1','p2','p3','p4','p5','g0','g1','g2')
NN_inputs = []
print "Neural N in struct = ", NN_inputs

#Productions
P = [ ('Design',           ['R']),                  \
#('Design',          ['R','C']),                     \
#('Design',          ['C','C','R']),                 \
('C',               ['Prom', 'Gene']),              \
('R',               ['Prom', 'g0']),                \
('Prom',            ['p1']),                        \
('Prom',            ['p2']),                        \
('Prom',            ['p3']),                        \
('Prom',            ['p4']),                        \
('Prom',            ['p5']),                        \
('Gene',            ['g1']),                        \
('Gene',            ['g2'])                         ]

language = parse_input.getLanguage(P,T)

print "Language = ", language
print "Num Designs = ", len(language)
print "Language = ", language
print "Longest design ", max(language, key=len)

############# GENERATE SINGLE CAS N.N. MODEL ############
start_time = time.time()
num_inputs = 2
num_hidden = 2
print 'num inputs = ', num_inputs
print 'num hidden units = ', num_hidden
print 'Compiling Model ... '
submodel = Sequential()
submodel.add(Dense(num_hidden, input_dim=num_inputs, W_regularizer=l1(0.1))) #, init='zero'))
submodel.add(Activation('relu'))
submodel.add(Dropout(0.3))
submodel.add(Dense(1))
submodel.add(Activation('sigmoid'))

rms = RMSprop()
sgd = SGD(lr=0.01)
submodel.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print 'Model compiled in {0} seconds'.format(time.time() - start_time)
print submodel.get_config()

############# RUN AGENT ############
############# SINGLE-CAS FORMAT DATA ############

X_train = []
Y_train = []
#cur_design = ['p3','g0']
#language.remove(cur_design)


testi = 0
langsize = int(len(language))
print "LANGSIZE = ", langsize
while testi<langsize:
	cur_design = language[testi]
	cur_char_data = lab.get_data(cur_design, testi)
	X_train, Y_train = format_data(cur_char_data, cur_design)
	# cur_design = random.choice(language)
	# language.remove(cur_design)
	testi += 1

print "\n\nX_train for full single cas language = ", X_train
print "\nY_train for full single cas language = ", Y_train

c = list(zip(X_train, Y_train))
random.shuffle(c)
X_train, Y_train = zip(*c)

print "\n\nShuffled X_train = ", X_train
print "\nShuffled Y_train = ", Y_train

Y_train_max = max(Y_train)[0]
Y_train_min = min(Y_train)[0]
print "\n\n Ymax ", Y_train_max
print "\n\n Ymin ", Y_train_min
Y_train_norm = []

for y in Y_train:
	Y_train_norm.append((y - Y_train_min) / (Y_train_max - Y_train_min))

X_train = np.array(X_train)


print "\n\nX_train = ", X_train
#print "\nY_train = ", np.array(Y_train)
Y_train = np.array(Y_train_norm)
print "\nNormalized Y_train = ", Y_train

# print "TRAINING DATA X: ", X_train
# print "Y: ", Y_train
Fin_xtrain = X_train[:len(X_train)/2]
Fin_xtest = X_train[len(X_train)/2:]

Fin_ytrain = Y_train[:len(Y_train)/2]
Fin_ytest = Y_train[len(Y_train)/2:]

print "\n\nFin_xtrain n test = ", Fin_xtrain, Fin_xtest

##################### TRAIN SINGLE CAS MODEL #######################
submodel.fit(Fin_xtrain, Fin_ytrain, nb_epoch=500, batch_size=5, verbose=2)
pred_goal_beh = submodel.predict(Fin_xtest)
print "MODEL PREDICTS FOR GOALD DES ", pred_goal_beh
print "Tru output is ", Fin_ytest
#print "Average of tru output = ", sum(Y_train)/len(Y_train)

# print "Weights = ", submodel.get_weights()
# submodel.save_weights("submod.h5")

################
#   Genes
################
#   1  2  3  4 
#   g0 g1 g2 g3
################

################
#   Proms
################
#   1  2  3  4 
#   p0 p1 p2 p3
################

##################
#      S
##################
#   1   2    3   4 
#   0 0.01  0.1  1
##################



#########################
#       p0 g0 0
#########################
#       1   2   3   4   5
#
# d[1]  1  -1  -1  -1  -1
#   s   1  -1  -1  -1  -1
#########################

########################
#       p0 g0 1
########################
#       1  2  3  4  5
#
# d[1]  1  0  0  0  0
#   s   0  0  0  0  1
########################

########################
#       p1 g0 1
########################
#       1  2  3  4  5
#
# d[1]  0  1  0  0  0
#   s   0  0  0  0  1
########################





