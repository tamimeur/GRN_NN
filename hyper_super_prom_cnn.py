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

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform

from keras.utils import np_utils


np.random.seed(1337)
X_train = []
Y_train = []

X_test = []
Y_test = []


############# LOAD DATA - NBT PAPER ########
pdata = pd.read_excel('nbt_sharon_pp.xlsx',header=0,parse_cols="C,G")


print "Is YO DATA BOI!!! \n", pdata
print pdata.columns
df = pdata[np.isfinite(pdata[u' expression'])]


############# Format NN inputs #############

def oneHotEncoder(seq):
	base_dict = {u'A':[1,0,0,0],u'C':[0,1,0,0],u'G':[0,0,1,0],u'T':[0,0,0,1]}
	return np.array([base_dict[x] for x in seq])

def oneHotDecoder(encseq):
	dec_seq = ""
	for x in encseq:
		if (x == np.array([1,0,0,0])).all():
			dec_seq += u'A'
		elif (x == np.array([0,1,0,0])).all():
			dec_seq += u'C'
		elif (x == np.array([0,0,1,0])).all():
			dec_seq += u'G'
		elif (x == np.array([0,0,0,1])).all():
			dec_seq += u'T'
	return dec_seq

X_data = np.empty([len(df),150,4])
indx = 0
for seq in df[u'sequence']:
	X_data[indx] = oneHotEncoder(seq)
	indx += 1

Y_data = np.array(df[[u' expression']])

print "CNN input \n", X_data
print "CNN output \n", Y_data


########## RANDOM TEST/TRAIN SPLIT #########
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.15, random_state=42)
norm_train = preprocessing.StandardScaler().fit_transform(y_train)
print "len X train: \n", len(X_train)
print "len Y train: \n", len(y_train)
print "len X test:  \n", len(X_test)
print "len Y test:  \n", len(y_test)

def data():
	return X_train, norm_train, X_test, y_test

############# TRAIN NN ############
def model(X_train,norm_train,X_test,y_test):
	model = Sequential()
	model.add(Convolution1D(nb_filter=30,filter_length=6,input_dim=4,input_length=150,border_mode="same", activation='relu'))
	model.add(Dropout({{uniform(0, 1)}}))
	model.add(Convolution1D(nb_filter=40,filter_length=6,input_dim=4,input_length=150,border_mode="same", activation='relu'))
	model.add(Flatten())
	model.add(Dense(40))
	model.add(Dropout({{uniform(0, 1)}}))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('linear'))

	#compile the model
	#adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	#model.compile(loss='mean_squared_error', optimizer=adam)
	rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
	model.compile(loss='mean_squared_logarithmic_error', optimizer=rms, metrics=['accuracy'])
	#print 'Model compiled in {0} seconds'.format(time.time() - start_time)

	#train the model
	model.fit(X_train, norm_train, batch_size={{choice([64,128])}}, nb_epoch=6, verbose=1)

	########### NN TRAINING RESULTS ##############

	norm_test = preprocessing.StandardScaler().fit_transform(y_test)
	score, acc = model.evaluate(X_test,norm_test)
	# predicted = model.predict(X_test)
	# slope, intercept, r_value, p_value, std_err = stats.linregress(norm_test.reshape(-1),predicted.reshape(-1))
	# print r_value**2
	return {'loss': -acc, 'status': STATUS_OK, 'model': model}

best_run, best_model = optim.minimize(model=model, data=data, algo=tpe.suggest, max_evals=5, trials=Trials())
print("Evalutation of best performing model:")
print(best_model.evaluate(X_test, Y_test))

predicted = best_model.predict(X_test)
slope, intercept, r_value, p_value, std_err = stats.linregress(norm_test.reshape(-1),predicted.reshape(-1))
print r_value**2
d = {'y_pred': predicted.reshape(-1), 'y_actual': norm_test.reshape(-1)}
res_df = pd.DataFrame(data=d)

sea.set(style="ticks", color_codes=True)
g = sea.JointGrid(predicted.reshape(-1),norm_test.reshape(-1))
g = g.plot_joint(plt.scatter, color='#9b59b6', edgecolor="white", alpha='0.1')

g.ax_joint.set_xticks([-1, 5, -1,0,1,2,3,4,5])
g.ax_joint.set_yticks([-1, 5, -1,0,1,2,3,4,5])
sea.plt.show()


######## FIND THE SEQ with MAX EXP #######
# from scipy.optimize import basinhopping

# ###### cos basinhopping example #######
# func = lambda x: np.cos(14.5 * x - 0.3) + (x + 0.2) * x
# x0=[1.]
# minimizer_kwargs = {"method": "BFGS"}
# ret = basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs, niter=200)
# print("global minimum: x = %.4f, f(x0) = %.4f" % (ret.x, ret.fun))
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
print "Predicted max indx: ", predicted.argmax()
print "Predicted max val: ", predicted[predicted.argmax()]

##########Creating lists for final generational promoter plot#################
fin_seq_indx = []
fin_exp = []
fin_gen = []

for prom in range(0,len(predicted)):
	fin_seq_indx.append(prom)
	fin_exp.append(predicted[prom][0])
	fin_gen.append(0)
###############################################################################



maxp = oneHotDecoder(X_test[predicted.argmax()])
print "Max promoter seq is: ", maxp
#GGGGACCAGGTGCCGTAAGCTGTTGAATGGCACTAAATCGGAACCCTAAAGGGAGCCCCCGATTTAGAGCTTGACGGCGGAAGACTCTCCTCCGGGCGCGGAAGACTCTCCTCCGCGGAAGACTCTCCTCCGGCGATCCTAGGGCGATCA

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

#GENERATION 1
new_promoters = []
start_seq = maxp

start_indx = 19
start_gen = 0
num_generations = 30
bases = [u'A',u'T',u'C',u'G']
#19 - 127 available indexes to start base mutations

for gen in range(0,num_generations):
	print " ======== GENERATION ", gen, "==========="
	for j in range(0,200):
		base_idx = np.random.randint(19,127)
		new_seq = list(start_seq)
		for k in range(0,5):
			new_seq[base_idx] = np.random.choice(bases)
			new_seq[base_idx+1] = np.random.choice(bases)
			new_seq[base_idx+2] = np.random.choice(bases)
			strnew_seq = "".join(new_seq)
			if df.loc[df[u'sequence'] == strnew_seq].empty:
				new_promoters.append(strnew_seq)

	print "New Promoters len ", len(new_promoters)

	##### format new promoter sequences as CNN input for prediction ####
	Z_test = np.empty([len(new_promoters),150,4])
	indx = 0
	for seq in new_promoters:
		Z_test[indx] = oneHotEncoder(seq)
		indx += 1

	Zpredicted = model.predict(Z_test)
	newp = new_promoters[Zpredicted.argmax()]
	print "Max exp of new designs = ", max(Zpredicted)
	print "Index of max exp prom = ", Zpredicted.argmax()
	print "Sequ of max prom = ", newp
	new_promoters = []
	start_seq = newp

	if(gen%3==0):
		###########################################
		for prom in range(0,len(Zpredicted)):
			fin_seq_indx.append(prom)
			fin_exp.append(Zpredicted.reshape(-1)[prom])
			fin_gen.append(gen)
		###########################################

print "len exp ", len(fin_exp)
print "len gen ", len(fin_gen)


sequence_variant=list(fin_seq_indx)
strength=list(fin_exp)
generation = list(fin_gen)

print "len expression_level ", len(strength)
print "len generation ", len(generation)

genpltdf = pd.DataFrame(dict(strength=strength, generation=generation))


print "df: ", genpltdf
genpl=sea.boxplot(x='generation',y='strength',data=genpltdf,hue='generation', width = 1.1)
axes = genpl.axes
axes.set_ylim(0,)
axes.set_aspect(1.5)
sea.plt.show()



