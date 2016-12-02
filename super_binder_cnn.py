#import grn_feedfwd as ffn
#import parse_input
#bandpass_simple4 as lab
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
import plotly.plotly as py


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

naive_protein = "ADPKKVLDKAKDRAENVVRKLKKELEELYKEARKLDLTQEMRDRIRLAAIAARIAAFGDIFHAIMEALEEARKLKKAGLVNSQQLDELKRRLEELDEEAAQRAEKLGKEFELKLEYG"

############# LOAD DATA - DY BINDING DATASET ########
rawdata = pd.read_excel('dy_binding_data.xlsx',header=0,parse_cols="L,M,C,D,E,F,G,H")


print "Is YO DATA BOI!!! \n", rawdata
print rawdata.columns

################ CLEAN/FORMAT DATA  #################
#maybe edit the mutation column in rawdata and later remove the position col
#ok make a function that passes this dataframe and returns a list of sequences, 
#then add the list to the df

def mutation_to_sequence(rawdf):
	protein_seq=[]
	indx = 0
	protein_seq.append(naive_protein)
	print rawdf.shape[0]
	# mut_seq = list(naive_protein)
	# print mut_seq
	# print mut_seq[rawdf[u'position'][1]]
	# mut_seq[rawdf[u'position'][1]] = rawdf[u'mutation'][1]
	# print mut_seq[rawdf[u'position'][1]]
	# print "length of naive prot = ", len(naive_protein)

	for indx in range(1,rawdf.shape[0]):
		mut_seq = list(naive_protein)
		mut_seq[(rawdf[u'position'][indx])-1] = str(rawdf[u'mutation'][indx])
		protein_seq.append(''.join(mut_seq))

	return protein_seq

seq_list = mutation_to_sequence(rawdata)
#print "List of mutations as sequences: ", seq_list

rawdata[u"sequence"]=pd.Series(seq_list).values

print "New raw data: \n ", rawdata

#shuffle data
rawdata = rawdata.iloc[np.random.permutation(len(rawdata))]

print "Shuffled raw data: \n ", rawdata


# print "len expression_level ", len(strength)
# print "len generation ", len(generation)

# genpltdf = pd.DataFrame(dict(strength=strength, generation=generation))
# print "df: ", genpltdf


#df = pdata[np.isfinite(pdata[u' expression'])]

# print "cleaned data \n", df
# print "length of cleaned data ", df.size
# print "length of sequence ", len(df[u'sequence'][1])

############# Format NN inputs #############

def oneHotEncoder(seq):
	base_dict = {'A':[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	'V':[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	'I':[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	'L':[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	'M':[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	'F':[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	'Y':[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	'W':[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
	'S':[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
	'T':[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
	'N':[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
	'Q':[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
	'C':[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
	'U':[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
	'G':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
	'P':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
	'R':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
	'H':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
	'K':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
	'D':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
	'E':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
	'*':[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]}
	return np.array([base_dict[x] for x in seq])

def oneHotDecoder(encseq):
	dec_seq = ""
	for x in encseq:
		if (x == np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'A'
		elif (x == np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'V'
		elif (x == np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'I'
		elif (x == np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'L'
		elif (x == np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'M'
		elif (x == np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'F'
		elif (x == np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'Y'
		elif (x == np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'W'
		elif (x == np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'S'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'T'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'N'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'Q'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'C'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0])).all():
			dec_seq += 'U'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0])).all():
			dec_seq += 'G'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0])).all():
			dec_seq += 'P'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0])).all():
			dec_seq += 'R'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0])).all():
			dec_seq += 'H'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0])).all():
			dec_seq += 'K'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])).all():
			dec_seq += 'D'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0])).all():
			dec_seq += 'E'
		elif (x == np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])).all():
			dec_seq += '*'
	return dec_seq
	#return np.array([decode_dict[x] for x in encseq])

X_data = np.empty([len(rawdata),117,22])

indx = 0
for seq in rawdata[u'sequence']:
	X_data[indx] = oneHotEncoder(seq)
	indx += 1
#fix
#Y_data = np.array(rawdata[[u'Bfl1']])
Y_data = np.empty([len(rawdata),6])
for i in range(0,rawdata.shape[0]):
	bfl1 = rawdata[u'Bfl1'][i] 
	bclb = rawdata[u'BclB'][i] 
	bcl2 = rawdata[u'Bcl2'][i] 
	bclxl = rawdata[u'BclXL'][i] 
	bclw = rawdata[u'BclW'][i] 
	mcl1 = rawdata[u'Mcl1'][i] 
	Y_data[i] = np.array([bfl1,bclb,bcl2,bclxl,bclw,mcl1])

print "CNN input \n", X_data
print "CNN output \n", Y_data


########## RANDOM TEST/TRAIN SPLIT #########
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.15, random_state=42)
#norm_train = preprocessing.StandardScaler().fit_transform(y_train)
print "len X train: \n", len(X_train)
print "len Y train: \n", len(y_train)
print "len X test:  \n", len(X_test)
print "len Y test:  \n", len(y_test)


############# TRAIN NN ############
#start_time = time.time()
model = Sequential()
model.add(Convolution1D(nb_filter=50,filter_length=6,input_dim=22,input_length=117,border_mode="same", activation='relu'))
model.add(Dropout(0.3))
model.add(Convolution1D(nb_filter=50,filter_length=6,input_dim=22,input_length=117,border_mode="same", activation='relu'))
#model.add(Dropout(0.3))
#model.add(Convolution1D(nb_filter=50,filter_length=6,input_dim=22,input_length=117,border_mode="same", activation='relu'))
model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='valid'))
model.add(Flatten())

model.add(Dense(40))
model.add(Dropout(0.2))
model.add(Activation('relu'))

model.add(Dense(6))
model.add(Activation('softmax'))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))


#compile the model
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
#model.compile(loss='mean_squared_error', optimizer=adam)
rms = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08)
#model.compile(loss='mean_squared_logarithmic_error', optimizer=rms)
model.compile(loss='categorical_crossentropy', optimizer=rms)
#model.compile(loss='mean_squared_logarithmic_error', optimizer=rms)
#print 'Model compiled in {0} seconds'.format(time.time() - start_time)

#train the model
#model.fit(X_train, norm_train, batch_size=32, nb_epoch=6, verbose=1)
model.fit(X_train, y_train, batch_size=128, nb_epoch=10, verbose=1)


########### NN TRAINING RESULTS ##############

norm_test = preprocessing.StandardScaler().fit_transform(y_test)
predicted = model.predict(X_test) #.reshape(-1)
print "TEST OUTPUTS: \n", y_test
print "PREDICTED: \n", predicted
#slope, intercept, r_value, p_value, std_err = stats.linregress(norm_test.reshape(-1),predicted.reshape(-1))

slope, intercept, r_value, p_value, std_err = stats.linregress(y_test.reshape(-1),predicted.reshape(-1))
print r_value**2

d = {'y_pred': predicted.reshape(-1), 'y_actual': y_test.reshape(-1)}
res_df = pd.DataFrame(data=d)

sea.set(style="ticks", color_codes=True)
g = sea.JointGrid(predicted.reshape(-1),y_test.reshape(-1)) #, xlim=(-3,3), ylim=(-3,3))
g = g.plot_joint(plt.scatter, color='#9b59b6', edgecolor="white", alpha='0.1')
# axes = g.axes
# axes.set_xlim(-1,5)
# axes.set_ylim(-1,5)
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
			if df.loc[df[u'sequence'] == strnew_seq].empty :
				#start_seq = strnew_seq
				new_promoters.append(strnew_seq)


	#print new_promoters
	print "New Promoters len ", len(new_promoters)

	##### format new promoter sequences as CNN input for prediction ####
	Z_test = np.empty([len(new_promoters),150,4])
	indx = 0
	for seq in new_promoters:
		Z_test[indx] = oneHotEncoder(seq)
		indx += 1

	#print "CNN input \n", Z_test, len(Z_test)
	Zpredicted = model.predict(Z_test)
	newp = new_promoters[Zpredicted.argmax()]
	print "Max exp of new designs = ", max(Zpredicted)
	print "Index of max exp prom = ", Zpredicted.argmax()
	print "Sequ of max prom = ", newp
	new_promoters = []
	start_seq = newp

	if(gen%3==0):
	#if(gen!=0 and gen%2==0):
	#if(gen!=0):
	#print "Zpredicted for gen ", gen, " = \n", Zpredicted.reshape(-1)
		################### #######################
		for prom in range(0,len(Zpredicted)):
			fin_seq_indx.append(prom)
			fin_exp.append(Zpredicted.reshape(-1)[prom])
			fin_gen.append(gen)
		###########################################

print "len exp ", len(fin_exp)
print "len gen ", len(fin_gen)



#print "FIN GEN PLOT LIST: \n",fin_seq_indx,"\n", fin_exp,"\n",fin_gen
sequence_variant=list(fin_seq_indx)
strength=list(fin_exp)
generation = list(fin_gen)

print "len expression_level ", len(strength)
print "len generation ", len(generation)

#genpltdf = pd.DataFrame(dict(sequence_variant=sequence_variant, expression_level=expression_level, generation=generation))
genpltdf = pd.DataFrame(dict(strength=strength, generation=generation))
#print "expression level list: ", expression_level
#print "generation list: ", generation
print "df: ", genpltdf

#sea.lmplot('sequence_variant','expression_level',data=genpltdf,hue='generation',fit_reg=False)

#sea.swarmplot(x='generation',y='expression_level',data=genpltdf,hue='generation')
#genpl=sea.violinplot(x='generation',y='expression_level',data=genpltdf,hue='generation', width = 7)
genpl=sea.boxplot(x='generation',y='strength',data=genpltdf,hue='generation', width = 1)
axes = genpl.axes
#axes.set_aspect(1.5)
axes.set_ylim(0,)
#axes.set_title('Predicted Promoter Strengths Across Generations of 3mer Mutations ')
sea.plt.show()

# N = num_generations / 2

# c = ['hsl('+str(h)+',50%'+',50%)' for h in linspace(0, 360, N)]

# data = [{
#     'y': expression_level, 
#     'type':'box',
#     'marker':{'color': c[i]}
#     } for i in range(int(N))]

# # format the layout
# layout = {'xaxis': {'showgrid':False,'zeroline':False, 'tickangle':60,'showticklabels':False},
#           'yaxis': {'zeroline':False,'gridcolor':'white'},
#           'paper_bgcolor': 'rgb(233,233,233)',
#           'plot_bgcolor': 'rgb(233,233,233)',
#           }

# py.iplot(data)


