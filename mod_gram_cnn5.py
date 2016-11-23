#import grn_feedfwd as ffn
import parse_input
import bandpass_simple4 as lab
import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import RMSprop, SGD, Adam
from keras.datasets import mnist
from keras.layers import Merge
from keras.regularizers import l1, l2, activity_l2
from keras.layers.convolutional import Convolution1D, MaxPooling1D
import random
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
import scipy.stats as stats
import seaborn as sea


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


print "Is YO DATA BOI!!! \n", pdata
#print "length of data ", pdata.size
print pdata.columns
df = pdata[np.isfinite(pdata[u' expression'])]

# print "cleaned data \n", df
# print "length of cleaned data ", df.size
# print "length of sequence ", len(df[u'sequence'][1])

############# Format NN inputs #############

def oneHotEncoder(seq):
	base_dict = {u'A':[1,0,0,0],u'C':[0,1,0,0],u'G':[0,0,1,0],u'T':[0,0,0,1]}
	return np.array([base_dict[x] for x in seq])
def oneHotDecoder(encseq):
	dec_seq = ""
	#decode_dict = {[1,0,0,0]:u'A',[0,1,0,0]:u'C',[0,0,1,0]:u'G',[0,0,0,1]:u'T'}
	for x in encseq:
		# print "This is x : ", x
		# np_test = np.array([0,1,0,0])
		# print "Here's an np conv: ", np_test
		# print "Testing equiv: ", (x==np_test).all()
		if (x == np.array([1,0,0,0])).all():
			dec_seq += u'A'
		elif (x == np.array([0,1,0,0])).all():
			dec_seq += u'C'
		elif (x == np.array([0,0,1,0])).all():
			dec_seq += u'G'
		elif (x == np.array([0,0,0,1])).all():
			dec_seq += u'T'
	return dec_seq
	#return np.array([decode_dict[x] for x in encseq])

X_data = np.empty([len(df),150,4])
indx = 0

Y_data = np.array(df[[u' expression']])

for seq in df[u'sequence']:
	X_data[indx] = oneHotEncoder(seq)
	indx += 1

print "CNN input \n", X_data
print "CNN output \n", Y_data


########## RANDOM TEST/TRAIN SPLIT #########
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.15, random_state=42)
norm_train = preprocessing.StandardScaler().fit_transform(y_train)
print "len X train: \n", len(X_train)
print "len Y train: \n", len(y_train)
print "len X test:  \n", len(X_test)
print "len Y test:  \n", len(y_test)


############# TRAIN NN ############
#start_time = time.time()
model = Sequential()
model.add(Convolution1D(nb_filter=30,filter_length=6,input_dim=4,input_length=150,border_mode="same", activation='relu'))
model.add(Dropout(0.1))
model.add(Convolution1D(nb_filter=40,filter_length=6,input_dim=4,input_length=150,border_mode="same", activation='relu'))

model.add(Flatten())

model.add(Dense(40))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('linear'))

#compile the model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#model.compile(loss='mean_squared_error', optimizer=adam)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
model.compile(loss='mean_squared_logarithmic_error', optimizer=rms)
#print 'Model compiled in {0} seconds'.format(time.time() - start_time)

#train the model
model.fit(X_train, norm_train, batch_size=128, nb_epoch=6, verbose=1)


########### RESULTS ##############

norm_test = preprocessing.StandardScaler().fit_transform(y_test)
predicted = model.predict(X_test) #.reshape(-1)
print "NORMED TEST: ", len(norm_test)
print "PREDICTED: ", len(predicted)
slope, intercept, r_value, p_value, std_err = stats.linregress(norm_test.reshape(-1),predicted.reshape(-1))
print r_value**2

d = {'y_pred': predicted.reshape(-1), 'y_actual': norm_test.reshape(-1)}
res_df = pd.DataFrame(data=d)

sea.set(style="ticks", color_codes=True)
g = sea.JointGrid(predicted.reshape(-1),norm_test.reshape(-1)) #, xlim=(-3,3), ylim=(-3,3))
g = g.plot_joint(plt.scatter, color='#9b59b6', edgecolor="white", alpha='0.1')
sea.plt.show()
#g = g.plot_marginals(sea.distplot, kde=False, color="#9b59b6")


######## FIND THE SEQ with MAX EXP #######

from scipy.optimize import basinhopping

###### cos basinhopping example #######
func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
x0=[1.]
minimizer_kwargs = {"method": "BFGS"}
ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs, niter=200)
print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))
#######################################


###### Find the max performing promoter ######
###### in cur set of data test and train ######
print "Searching for max prom: "
print df[[u' expression']]
maxindx=df[[u' expression']].idxmax()
print "TRUE max index ", maxindx
print "TRUE max value ", df.ix[maxindx]
print "TRUE max sequence ", df[[u'sequence']].ix[maxindx]

print "Searching for max prom in normed vals: "
print "Norm test max indx: ", norm_test.argmax()
print "Norm test max val: ", norm_test[norm_test.argmax()]
print "Norm train max indx: ", norm_train.argmax()
print "Norm train max val: ", norm_train[norm_train.argmax()]

# maxp = max(norm_test[norm_test.argmax()],norm_train[norm_train.argmax()])
maxp = oneHotDecoder(X_test[norm_test.argmax()])
print "Max promoter seq is: ", maxp
### sadly the normed max doesn't match the true max from the data set, but whatever I guess.
### OH! It actually is! There may be more than one promoter seq that is the same max val :)
### we'll still start with the seq that is the normed max of the data set.

### now we need to vary 3mers and create new variants that don't already exist in the original dataset
print "where is max normed prom in orig data set? ", df.loc[df[u'sequence'] == maxp]

fake_isin_test = ""
for i in range(0,150):
	fake_isin_test += u'A'

print len(fake_isin_test)
print len(maxp)

print "Check that fake_isin_test is not in original data set ", df.loc[df[u'sequence'] == fake_isin_test].empty

new_promoters = []

# def gen_new_seq(parent_seq, generation, fixed_child_num, base_indx):
# 	bases = [u'A',u'T',u'C',u'G']


start_seq = maxp
start_indx = 19
start_gen = 0
num_child = 1
bases = [u'A',u'T',u'C',u'G']

# for j in range(0,10):
# 	new_promoters.append(gen_new_seq(start_seq, start_gen, num_child, start_indx))

# j = start_indx
# while j < len(start_seq):
# 	new_prom = start_seq

# 	new_prom[j] = random.choice(bases)
# 	new_promoters.append[]
# 	j++


