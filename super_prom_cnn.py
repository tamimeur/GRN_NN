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


########### NN TRAINING RESULTS ##############

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
# axes = g.axes
# axes.set_xlim(-1,5)
# axes.set_ylim(-1,5)
g.ax_joint.set_xticks([-1, 5, -1,0,1,2,3,4,5])
g.ax_joint.set_yticks([-1, 5, -1,0,1,2,3,4,5])
sea.plt.show()
#g = g.plot_marginals(sea.distplot, kde=False, color="#9b59b6")

########## FOR PLOTTING A HEATMAP ###############
#first find a sequence that's exp level is the median (0)
#to do this I make a copy of norm_test ... make all values abs(norm_test)
#and then take the min expression. But I need to keep track 
#of the x_test it's related to because that sequence will be our base seq
#from which all mutations in the heat map will be relative to.

norm_test_abs = np.copy(norm_test)
np.absolute(norm_test)
min_indx = np.argmin(norm_test)
base_HM_val = norm_test[min_indx]
print "Minimut NORM TEST VALUE ", base_HM_val
base_HM_seq = list(str(oneHotDecoder(X_test[min_indx])))
print "Seq with this exp val is: ", base_HM_seq
sub_base_HM_seq = list(base_HM_seq[19:127])
start_base_HM_seq = list(base_HM_seq[0:19])
end_base_HM_seq = list(base_HM_seq[127:(len(base_HM_seq))])

print "Lengths base, start, end seq's: ", len(sub_base_HM_seq), len(start_base_HM_seq), len(end_base_HM_seq)

# print "start chunk of the seq: ", start_base_HM_seq
# print "middle chunk of the seq: ", sub_base_HM_seq
# print "end chunk of the seq: ", end_base_HM_seq
# sub_base_HM_seq.extend(end_base_HM_seq)
# start_base_HM_seq.extend(sub_base_HM_seq)

# print "rebuilt seq: ", start_base_HM_seq
# print "\n\n\n"
# ##GGGGACCAGGTGCCGTAAGACTCACGGGCGCACTAAATCGGAACCCTAAAGGGAGCCGGAACCGGCGAGCTTGACGGGGAAAGCCATGTAATTACC|T|AATAGGGAAATTTACACGATTTTTTTTTTTTTTTAGCGATCCTAGGGCGATCA##
# print "test of test (expect True) : ", ['a','b','c'] == ['a','b','c']
# print "test of test (expect False) : ", ['a','d','c'] == ['a','b','c']

# print "Test splitting of seq (expect True) : ", base_HM_seq == start_base_HM_seq, len(base_HM_seq), len(start_base_HM_seq)

pos_HM = []
mut_HM = []
val_HM = []

base_seq_indx = 0

for base in sub_base_HM_seq:
	bases = ['A','C','T','G']
	bases.remove(base)
	for mut in bases:
		pos_HM.append(base_seq_indx)
		mut_HM.append(mut)
		mut_seq = list(sub_base_HM_seq)
		mut_seq[base_seq_indx] = mut
		
		cpy_mut_seq = list(mut_seq)
		cpy_start_base_HM_seq = list(start_base_HM_seq)
		cpy_end_base_HM_seq = list(end_base_HM_seq)

		#print "Lengths base, start, end seq's: ", len(cpy_mut_seq), len(cpy_start_base_HM_seq), len(cpy_end_base_HM_seq)

		cpy_mut_seq.extend(cpy_end_base_HM_seq)
		cpy_start_base_HM_seq.extend(cpy_mut_seq)

		mut_seq = list(cpy_start_base_HM_seq)
		#print "LENGTH CHECK OF MUT SEQ: ",len(mut_seq), len(base_HM_seq) 
		X_mutdata = np.empty([1,150,4])
		X_mutdata[0] = oneHotEncoder(mut_seq)
		print "MUT SEQ: ", str(mut_seq)
		mut_exp = model.predict(X_mutdata)
		print "Pred EXP: ", mut_exp
		val_HM.append((mut_exp - base_HM_val).reshape(-1)[0])
		print "Pred VAL: ", mut_exp - base_HM_val, "\n"
	base_seq_indx += 1

print "################ HEAT MAP LISTS ################\n\n\n"
print "\nposHM: \n", pos_HM
print "\nmutHM: \n", mut_HM
print "\nvalHM: \n", val_HM

genHMdf = pd.DataFrame({'position': pos_HM, 'mutation': mut_HM, 'value': val_HM})
result = genHMdf.pivot(index='position', columns='mutation', values='value')
ax = sea.heatmap(result)
sea.plt.show()

#now generate a full library of single point mutations off of this base_HM_seq
#Y : position
#X : mutation
#val : NN exp output for the seq - exp of base_HM_seq

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
#print "where is median normed prom in orig data set? (we just need one) ", df.loc[df[u' expression'] == 0]
#print "JUST FOR KICKS: \n", df

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
genpl=sea.boxplot(x='generation',y='strength',data=genpltdf,hue='generation', width = 1.1)
axes = genpl.axes
axes.set_ylim(0,)
axes.set_aspect(1.5)
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


