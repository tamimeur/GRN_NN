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

############# GENERATE LANGUAGE ############

#The Grammar
Signals = ('s')
T = ('p1','p2','p3','g0','g1','g2')
NN_inputs = ['p1','p2','p3','g0','g1','g2']
print "Neuaral N in struct = ", NN_inputs

#Productions
P = [ ('Design',           ['R']),                  \
('Design',          ['C','R']),                     \
('Design',          ['C','C','R']),                 \
('C',               ['Prom', 'Gene']),              \
('R',               ['Prom', 'g0']),                \
('Prom',            ['p1']),                        \
('Prom',            ['p2']),                        \
('Prom',            ['p3']),                        \
('Gene',            ['g1']),                        \
('Gene',            ['g2'])                         ]

language = parse_input.getLanguage(P,T)

print "Language = ", language
print "Num Designs = ", len(language)
pred_design = ['p3','g2','p2','g1','p1','g0']
language.remove(pred_design)
print "Language = ", language
print "Num Designs = ", len(language)

############# GENERATE N.N. MODEL ############
start_time = time.time()
num_inputs = len(T) + len(Signals)
num_hidden = num_inputs + (num_inputs/2)
print 'num inputs = ', num_inputs
print 'num hidden units = ', num_hidden
print 'Compiling Model ... '
model = Sequential()
model.add(Dense(num_hidden, input_dim=num_inputs)) #, init='zero'))
model.add(Activation('relu'))
#model.add(Dropout(0.4))
model.add(Dense(1))
model.add(Activation('relu'))

rms = RMSprop()
#model.compile(loss='sparse_categorical_crossentropy', optimizer=rms)
sgd = SGD(lr=0.001)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
print 'Model compield in {0} seconds'.format(time.time() - start_time)


############# RUN AGENT ############
#cur_design = ['p1','g0']
def format_data(char_data, des):
	global NN_inputs
	global X_train
	global Y_train
	# X_train = []
	# Y_train = []
	
	inducer_conc = char_data[0]
	res_conc = char_data[1]

	parts_in_list = []
	for part in NN_inputs:
		if part in des:
			parts_in_list.append(1)
		else:
			parts_in_list.append(0)

	for inducer in inducer_conc:
		new_list = list(parts_in_list)
		new_list.append(inducer)
		X_train.append(new_list)

	for val in res_conc:
		Y_train.append([val])

	
	# print "X_train = ", X_train
	# print "\n"
	# print "Y_train = ", Y_train
	return X_train, Y_train

#cur_design = ['p3','g2','p2','g1','p1','g0']
cur_design = ['p3','g0']
language.remove(cur_design)


# traini = 0
# while traini<(len(language)/2):
# 	#print "Training with design ", cur_design
# 	cur_char_data = lab.get_data(cur_design, traini)
# 	X_train, Y_train = format_data(cur_char_data, cur_design)
	
# 	# train_res = model.train_on_batch(X_train, Y_train)
# 	# print "\n"
# 	# #print model.metrics_names
# 	# print "Batch ", testi, " training"
# 	# print "Loss ... ", train_res[0]
# 	# print "Accuracy ... ", train_res[1]
# 	cur_design = random.choice(language)
# 	language.remove(cur_design)
# 	traini += 1
# final_X_train = list(X_train)
# final_Y_train = list(Y_train)


# testi = 0
# X_train = []
# Y_train = []
# while testi<(len(language)):
# 	#print "Training with design ", cur_design
# 	cur_char_data = lab.get_data(cur_design, testi)
# 	X_train, Y_train = format_data(cur_char_data, cur_design)
	
# 	# train_res = model.train_on_batch(X_train, Y_train)
# 	# print "\n"
# 	# #print model.metrics_names
# 	# print "Batch ", testi, " training"
# 	# print "Loss ... ", train_res[0]
# 	# print "Accuracy ... ", train_res[1]
# 	cur_design = random.choice(language)
# 	language.remove(cur_design)
# 	testi += 1

# final_X_test = list(X_train)
# final_Y_test = list(Y_train)


# print "Xtrain = ", final_X_train
# print "Ytrain = ", final_Y_train
# print "Xtest = ", final_X_test
# print "Ytest = ", final_Y_test

# history = LossHistory()

# model.fit(final_X_train, final_Y_train, nb_epoch=25, batch_size=5, verbose=2, 
# 	callbacks=[history], validation_data=(final_X_test, final_Y_test),)


# #PREDICT GOAL DES
# cur_char_data = lab.get_data(pred_design, testi)
# X_train = []
# Y_train = []
# X_train, Y_train = format_data(cur_char_data, cur_design)
# pred_goal_beh = model.predict(X_train,Y_train)
# print "MODEL PREDICTS FOR GOALD DES ", pred_goal_beh

X_train = []
Y_train = []
testi = 0
while testi<len(language):
	#print "Training with design ", cur_design
	cur_char_data = lab.get_data(cur_design, testi)
	X_train, Y_train = format_data(cur_char_data, cur_design)
	#model.fit(X_train, Y_train, nb_epoch=10, batch_size=5, verbose=2)
	# train_res = model.train_on_batch(X_train, Y_train)
	# print "\n"
	# #print model.metrics_names
	# print "Batch ", testi, " training"
	# print "Loss ... ", train_res[0]
	# print "Accuracy ... ", train_res[1]
	cur_design = random.choice(language)
	language.remove(cur_design)
	testi += 1
print len(X_train), len(Y_train)
model.fit(X_train, Y_train, nb_epoch=5000, batch_size=5, verbose=1)

#PREDICT GOAL DES
cur_char_data = lab.get_data(pred_design, testi)
X_train = []
Y_train = []
X_train, Y_train = format_data(cur_char_data, pred_design)
print X_train, Y_train
pred_goal_beh = model.predict(X_train)
print "MODEL PREDICTS FOR GOALD DES ", pred_goal_beh


# testi = 0
# while testi<30:
# 	print "Training with design ", cur_design
# 	cur_char_data = lab.get_data(cur_design, testi)
# 	X_train, Y_train = format_data(cur_char_data, cur_design)
# 	model.fit(X_train, Y_train, nb_epoch=10, batch_size=5, verbose=2)
# 	# train_res = model.train_on_batch(X_train, Y_train)
# 	# print "\n"
# 	# #print model.metrics_names
# 	# print "Batch ", testi, " training"
# 	# print "Loss ... ", train_res[0]
# 	# print "Accuracy ... ", train_res[1]
# 	cur_design = random.choice(language)
# 	language.remove(cur_design)
# 	testi += 1

