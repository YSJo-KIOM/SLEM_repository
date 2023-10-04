from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Input, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.callbacks import LambdaCallback
from keras import backend as K
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from multiprocessing import Process, Queue, Manager
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, getopt, os, random
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CustomConnected(Dense): # The class for sparse connected layer
	def __init__(self,units,connections,**kwargs):

		#this is matrix A
		self.connections = connections

		#initalize the original Dense with all the usual arguments
		super(CustomConnected,self).__init__(units,**kwargs)
	def call(self, inputs):
		output = K.dot(inputs, self.kernel * self.connections)
		if self.use_bias:
			output = K.bias_add(output, self.bias)
		if self.activation is not None:
			output = self.activation(output)
		return output

# The function for model initialization
def create_model(l_rate, linear, rand, w_dist, conn_thr, W1, W2, W3, B1=None, B2=None, B3=None):
	num_in = W1.shape[0]; num_L1 = W1.shape[1]; num_L2 = W2.shape[1]; num_L3 = W3.shape[1];
	if linear == False:
		#print('==non-linear mode==')
		act = 'tanh'
	else:
		#print('==linear mode==')
		act = 'relu'

	#print(w_dist)
	#edge_thrs_dict = {'he':[0.24, 0.41, 0.76], 'xv':[0.2, 0.32, 0.75]} thr = 0.05

	# Default connection thresholds for each layer (separately evaluated)
	edge_thrs_dict = {}
	edge_thrs_dict['0.1'] = {'he':[0.19, 0.35, 0.6], 'xv':[0.16, 0.27, 0.37]} # thr = 0.1
	edge_thrs_dict['0.15'] = {'he':[0.19, 0.35, 0.6], 'xv':[0.14, 0.23, 0.3]}
	edge_thrs_dict['0.2'] = {'he':[0.19, 0.35, 0.6], 'xv':[0.12, 0.21, 0.26]}
	edge_thrs_dict['0.25'] = {'he':[0.19, 0.35, 0.6], 'xv':[0.11, 0.19, 0.24]}
	edge_thrs_dict['0.3'] = {'he':[0.19, 0.35, 0.6], 'xv':[0.1, 0.17, 0.22]}
	edge_thrs_dict['0.35'] = {'he':[0.19, 0.35, 0.6], 'xv':[0.09, 0.15, 0.2]}
	edge_thrs_dict['0.4'] = {'he':[0.19, 0.35, 0.6], 'xv':[0.08, 0.14, 0.18]}

	# Connection thresholds by the provided file
	edge_thrs = edge_thrs_dict[conn_thr][w_dist]
	model = Sequential()
	W1_conn = np.array(W1); W2_conn = np.array(W2); W3_conn = np.array(W3); W1_init = np.array(W1); W2_init = np.array(W2); W3_init = np.array(W3);
	W1_conn[abs(W1_conn)<edge_thrs[0]]=0; W2_conn[abs(W2_conn)<edge_thrs[1]]=0; W1_conn[abs(W1_conn)>=edge_thrs[0]]=1; W2_conn[abs(W2_conn)>=edge_thrs[1]]=1;
	W3_conn[abs(W3_conn)<edge_thrs[2]]=0; W3_conn[abs(W3_conn)>=edge_thrs[2]]=1;
	W1_init[abs(W1_init)<edge_thrs[0]]=0; W2_init[abs(W2_init)<edge_thrs[1]]=0; W3_init[abs(W3_init)<edge_thrs[2]]=0;


	if rand == 0: # Initialization by input weights
		np.random.seed(random.randint(0,100000))
		nz_W1 = np.count_nonzero(W1_conn); nz_W2 = np.count_nonzero(W2_conn); nz_W3 = np.count_nonzero(W3_conn);
		W1_conn = np.zeros(num_in*num_L1); W2_conn = np.zeros(num_L1*num_L2); W3_conn = np.zeros(num_L2*num_L3);
		W1_conn[:nz_W1] = 1; W2_conn[:nz_W2] = 1; W3_conn[:nz_W3] = 1;
		np.random.shuffle(W1_conn); np.random.shuffle(W2_conn); np.random.shuffle(W3_conn);
		W1_conn = W1_conn.reshape(num_in,num_L1); W2_conn = W2_conn.reshape(num_L1,num_L2); W3_conn = W3_conn.reshape(num_L2,num_L3);
		if w_dist == 'xv':
			W1_init = np.random.normal(0, np.sqrt(2.0/num_in),(num_in,num_L1)); W2_init = np.random.normal(0, np.sqrt(2.0/num_L1),(num_L1,num_L2));
			W3_init = np.random.normal(0, np.sqrt(2.0/num_L2),(num_L2,num_L3));
		else:
			W1_init = np.random.normal(0, np.sqrt(2.0/num_in),(num_in,num_L1)); W2_init = np.random.normal(0, np.sqrt(2.0/num_L1),(num_L1,num_L2));
			W3_init = np.random.normal(0, np.sqrt(2.0/num_L2),(num_L2,num_L3));

		if len(B1) == 3:
			#print('Set Initial Weights')
			#print('=========')

			W1_init = W1_conn * W1_init; W2_init = W2_conn * W2_init; W3_init = W3_conn * W3_init;

			#print('==initial==')
			model.layers[0].set_weights([W1_init,model.layers[0].get_weights()[1]])
			model.layers[2].set_weights([W2_init,model.layers[2].get_weights()[1]])
			model.layers[4].set_weights([W3_init,model.layers[4].get_weights()[1]])
		else:
			model.layers[0].set_weights([W1,B1])
			model.layers[2].set_weights([W2,B2])
			model.layers[4].set_weights([W3,B3])

	if rand == 1: # Random initialization
		model.add(Dense(units=num_L1, input_shape=(num_in,), kernel_initializer='glorot_normal'))
		model.add(Activation(act))
		model.add(Dense(units=num_L2, kernel_initializer='glorot_normal'))
		model.add(Activation(act))
		model.add(Dense(units=num_L3, kernel_initializer='glorot_normal'))
		model.add(Activation(act))
		model.add(Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal'))
	else:
		init_dict = {'he':'he_normal', 'xv':'glorot_normal'}
		model.add(CustomConnected(connections = W1_conn, units=num_L1, input_shape=(num_in,), kernel_initializer=init_dict[w_dist]))
		#model.add(BatchNormalization())
		model.add(Activation(act))
		model.add(CustomConnected(connections = W2_conn, units=num_L2, kernel_initializer=init_dict[w_dist]))
		#model.add(BatchNormalization())
		model.add(Activation(act))
		model.add(CustomConnected(connections = W3_conn, units=num_L3, kernel_initializer=init_dict[w_dist]))
		#model.add(BatchNormalization())
		model.add(Activation(act))
		model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init_dict[w_dist]))

		if len(B1) == 3:
			#print('Set Initial Weights')
			#print('=========')

			W1_init = W1_conn * W1_init; W2_init = W2_conn * W2_init; W3_init = W3_conn * W3_init;

			#print('==initial==')
			model.layers[0].set_weights([W1_init,model.layers[0].get_weights()[1]])
			model.layers[2].set_weights([W2_init,model.layers[2].get_weights()[1]])
			model.layers[4].set_weights([W3_init,model.layers[4].get_weights()[1]])
		else:
			model.layers[0].set_weights([W1,B1])
			model.layers[2].set_weights([W2,B2])
			model.layers[4].set_weights([W3,B3])

	myadam = Adam(learning_rate=l_rate)
	model.compile(loss='binary_crossentropy', optimizer=myadam, metrics=['accuracy', tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.TruePositives(name='true_positives'),tf.keras.metrics.TrueNegatives(name='true_negatives'),tf.keras.metrics.FalsePositives(name='false_positives'),tf.keras.metrics.FalseNegatives(name='false_negatives')])
	#model.summary()
	return model

# The function for loading existing model
def load_model(l_rate, linear, rand, w_dist, W1, W2, W3, W4, B1=None, B2=None, B3=None, B4=None):
	num_in = W1.shape[0]; num_L1 = W1.shape[1]; num_L2 = W2.shape[1]; num_L3 = W3.shape[1];
	if linear == False:
		#print('==non-linear mode==')
		act = 'tanh'
	else:
		#print('==linear mode==')
		act = 'relu'
	w_dist = 'xv'
	model = Sequential()
	W1_conn = np.array(W1); W2_conn = np.array(W2); W3_conn = np.array(W3); W1_init = np.array(W1); W2_init = np.array(W2); W3_init = np.array(W3);
	W1_conn[abs(W1_conn)>0]=1; W2_conn[abs(W2_conn)>0]=1; W3_conn[abs(W3_conn)>0]=1;

	init_dict = {'he':'he_normal', 'xv':'glorot_normal'}

	if rand == True:
		model.add(Dense(units=num_L1, input_shape=(num_in,), kernel_initializer='glorot_normal'))
		model.add(Activation(act))
		model.add(Dense(units=num_L2, kernel_initializer='glorot_normal'))
		model.add(Activation(act))
		model.add(Dense(units=num_L3, kernel_initializer='glorot_normal'))
		model.add(Activation(act))
		model.add(Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal'))
	else:
		model.add(CustomConnected(connections = W1_conn, units=num_L1, input_shape=(num_in,), kernel_initializer=init_dict[w_dist]))
		model.add(Activation(act))
		model.add(CustomConnected(connections = W2_conn, units=num_L2, kernel_initializer=init_dict[w_dist]))
		model.add(Activation(act))
		model.add(CustomConnected(connections = W3_conn, units=num_L3, kernel_initializer=init_dict[w_dist]))
		model.add(Activation(act))
		model.add(Dense(units=1, activation='sigmoid', kernel_initializer=init_dict[w_dist]))

		W1_init = W1_conn * W1_init; W2_init = W2_conn * W2_init; W3_init = W3_conn * W3_init;

	if len(B1) == 3:
		model.layers[0].set_weights([W1_init,model.layers[0].get_weights()[1]])
		model.layers[2].set_weights([W2_init,model.layers[2].get_weights()[1]])
		model.layers[4].set_weights([W3_init,model.layers[4].get_weights()[1]])
		model.layers[6].set_weights([W4,model.layers[6].get_weights()[1]])
	else:
		#print('===Load weights===')
		model.layers[0].set_weights([W1_init,B1])
		model.layers[2].set_weights([W2_init,B2])
		model.layers[4].set_weights([W3_init,B3])
		model.layers[6].set_weights([W4,B4])

	myadam = Adam(learning_rate=l_rate)

	model.compile(loss='binary_crossentropy', optimizer=myadam,metrics=['accuracy',tf.keras.metrics.Precision(),tf.keras.metrics.Recall(),tf.keras.metrics.TrueNegatives(name='true_negatives'),tf.keras.metrics.FalsePositives(name='false_positives'),tf.keras.metrics.FalseNegatives(name='false_negatives')])
	#model.summary()
	return model

def kerasMetric(Y_t,Y_p,metric):
	if metric == 'TP': met = tf.keras.metrics.TruePositives();
	if metric == 'TN': met = tf.keras.metrics.TrueNegatives();
	if metric == 'FP': met = tf.keras.metrics.FalsePositives();
	if metric == 'FN': met = tf.keras.metrics.FalseNegatives();
	if metric == 'AUROC': met = tf.keras.metrics.AUC(curve='ROC');
	if metric == 'AUPR': met = tf.keras.metrics.AUC(curve='PR');
	met.update_state(Y_t,Y_p)
	ret = met.result().numpy()
	return ret

# The function for model evaluation
def model_test(X_tr, Y_tr, X_te, Y_te, W1, W2, W3, B1, B2, B3, batch, epo, l_rate, rand, linear, w_dist, conn_thr, out_list):
	model = create_model(l_rate,linear, rand, w_dist, conn_thr, W1, W2, W3, B1, B2, B3)
	#print('===')
	#print(model.layers[4].get_weights()[0][3])
	#print_weights1 = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[0].get_weights()[0][0]))
	#print_weights2 = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[2].get_weights()[0][0]))
	#print_weights3 = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[4].get_weights()[0][0]))
	hist = model.fit(X_tr, Y_tr, epochs = epo, batch_size=batch, verbose = 0)#, callbacks = [print_weights1, print_weights2,print_weights3])
	Y_pred = model.predict(X_te)
	Y_pred = Y_pred.reshape((Y_pred.shape[0],))
	TP = kerasMetric(Y_te, Y_pred,'TP')
	TN = kerasMetric(Y_te, Y_pred,'TN')
	FP = kerasMetric(Y_te, Y_pred,'FP')
	FN = kerasMetric(Y_te, Y_pred,'FN')
	AUROC = kerasMetric(Y_te, Y_pred,'AUROC')
	AUPR = kerasMetric(Y_te, Y_pred,'AUPR')
	#print(TP,TN,FP,FN,AUROC, AUPR)
	scores = model.evaluate(X_te, Y_te)
	#print(scores[2])
	txt_form = str(scores[1]) + ','+str(TP)+','+str(TN)+','+str(FP)+','+str(FN)+','+str(AUROC)+','+str(AUPR)
	out_list['acc']+=[(scores[1])]
	out_list['P']+=[(scores[2])]
	out_list['R']+=[(scores[3])]
	out_list['raw'] += [txt_form]
	return None


# The function for saving model
def model_save(X_tr, Y_tr, X_te, Y_te, W1, W2, W3, B1, B2, B3, batch, epo, l_rate, rand, linear, w_dist, conn_thr, out_path, features):
	model = create_model(l_rate,linear, rand, w_dist, conn_thr, W1, W2, W3, B1, B2, B3)
	print_weights1 = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[0].get_weights()[0]))
	print_weights2 = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[2].get_weights()[0]))
	print_weights3 = LambdaCallback(on_epoch_end=lambda epoch, logs: print(model.layers[4].get_weights()[0]))
	hist = model.fit(X_tr, Y_tr, epochs = epo, batch_size=batch, verbose = 0)#, callbacks = [print_weights1, print_weights2])

	scores = model.evaluate(X_te, Y_te)
	print(scores)

	w1_out = pd.DataFrame(model.layers[0].get_weights()[0],index = features[0],columns=features[1])
	w2_out = pd.DataFrame(model.layers[2].get_weights()[0],index = features[1],columns=features[2])
	w3_out = pd.DataFrame(model.layers[4].get_weights()[0],index = features[2],columns=features[3])
	w4_out = pd.DataFrame(model.layers[6].get_weights()[0],index = features[3])
	b1_out = pd.DataFrame(model.layers[0].get_weights()[1],index = features[1])
	b2_out = pd.DataFrame(model.layers[2].get_weights()[1],index = features[2])
	b3_out = pd.DataFrame(model.layers[4].get_weights()[1],index = features[3])
	b4_out = pd.DataFrame(model.layers[6].get_weights()[1])
	w1_out.to_csv(out_path.split('.')[0]+'_W1.txt', sep ='\t')
	w2_out.to_csv(out_path.split('.')[0]+'_W2.txt', sep ='\t')
	w3_out.to_csv(out_path.split('.')[0]+'_W3.txt', sep ='\t')
	w4_out.to_csv(out_path.split('.')[0]+'_W4.txt', sep ='\t')
	b1_out.to_csv(out_path.split('.')[0]+'_B1.txt', sep ='\t')
	b2_out.to_csv(out_path.split('.')[0]+'_B2.txt', sep ='\t')
	b3_out.to_csv(out_path.split('.')[0]+'_B3.txt', sep ='\t')
	b4_out.to_csv(out_path.split('.')[0]+'_B4.txt', sep ='\t')
	return None

def test_eval(X, Y, W1, W2, W3, W4, B1, B2, B3, B4, batch, epo, l_rate, rand, linear, w_dist):
	model = load_model(l_rate, linear, rand, w_dist, W1, W2, W3, W4, B1, B2, B3, B4)

	return model.evaluate(X,Y)[1]

def cv_test(X, Y, W1, W2, W3, W4, B1, B2, B3, B4, batch, epo, l_rate, rand, linear, w_dist, out_list):
	model = load_model(l_rate, linear, rand, w_dist, W1, W2, W3, W4, B1, B2, B3, B4)
	score = model.evaluate(X,Y)
	print(score)
	Y_pred = model.predict(X)
	#print(Y)
	print(Y_pred)
	out_list['acc']+=[(score[1])]
	out_list['P']+=[(score[2])]
	out_list['R']+=[(score[3])]
	return None

# The function for reading model files
def readPaths(w1_path, w2_path, w3_path, w4_path, b1_path, b2_path, b3_path, b4_path):
	w1_data = pd.read_csv(w1_path,sep='\t',index_col=0)
	w2_data = pd.read_csv(w2_path,sep='\t',index_col=0)
	w3_data = pd.read_csv(w3_path,sep='\t',index_col=0)
	w4_data = pd.read_csv(w4_path,sep='\t',index_col=0)

	b1_data = pd.read_csv(b1_path,sep='\t',index_col=0)
	b2_data = pd.read_csv(b2_path,sep='\t',index_col=0)
	b3_data = pd.read_csv(b3_path,sep='\t',index_col=0)
	b4_data = pd.read_csv(b4_path,sep='\t',index_col=0)

	B1 = b1_data.values.flatten()
	B2 = b2_data.values.flatten()
	B3 = b3_data.values.flatten()
	B4 = b4_data.values.flatten()

	in_features = w1_data.index
	L1_features = w1_data.columns
	L2_features = w2_data.columns
	L3_features = w3_data.columns

	W1 = w1_data[L1_features].values
	W2 = w2_data[L2_features].values
	W3 = w3_data[L3_features].values
	W4 = w4_data.values

	return (W1, W2, W3, W4, B1, B2, B3, B4)

def sampling(data):
	SZ_num = data[data['Schizo']==1].shape[0]
	ctrl_num = data[data['Schizo']==0].shape[0]

	if ctrl_num > SZ_num:
		SZ = data[data['Schizo']==1]
		ctrl_id = np.random.choice(data[data['Schizo']==0].index,size=SZ_num)
		ctrl = data.loc[ctrl_id]
	else:
		ctrl = data[data['Schizo']==0]
		SZ_id = np.random.choice(data[data['Schizo']==1].index,size=ctrl_num)
		SZ = data.loc[SZ_id]

	out_data = pd.concat([SZ,ctrl])
	return out_data

def case_sampling(data):
	out_data = data[data['Schizo']==1]
	return out_data

def main():
	opts, args = getopt.getopt(sys.argv[1::],"ri:o:w:s:p:m:",["w1=","w2=","w3=","w4=","b1=","b2=","b3=","b4=","pr=","y1=","y2=","y3=","y4=" "model="])

	in_path = None; out_path = None; w1_path = None; w2_path = None; w3_path = None; w4_path = None; seed = 0; rand = False;
	b1_path = None; b2_path = None; b3_path = None; b4_path = None; params = None; linear = False; mode = None;
	y1_path = None; y2_path = None; y3_path = None; y3_path = None; w_path = None; prefix = None; r_path = None; w_dist = None;

	for o, a in opts:
			if o == "-i": in_path = a;
			if o == "-o": out_path = a;
			if o == '-w': w_path = a;
			if o == '-p': r_path = a;
			if o == "--w1": w1_path = a;
			if o == "--w2": w2_path = a;
			if o == "--w3": w3_path = a;
			if o == "--w4": w4_path = a;
			if o == "--y1": y1_path = a;
			if o == "--y2": y2_path = a;
			if o == "--y3": y3_path = a;
			if o == "--y4": y4_path = a;
			if o == "-s": seed = int(a);
			if o == "-r": rand = True;
			if o == "-m": mode = a;
			if o == "--b1": b1_path = a;
			if o == "--b2": b2_path = a;
			if o == "--b3": b3_path = a;
			if o == "--b4": b4_path = a;
			if o == "--pr": params = a.split(',');
			if o == "--model": prefix = a;


	if prefix != None:
		w1_path = prefix + '_W1.txt'
		w2_path = prefix + '_W2.txt'
		w3_path = prefix + '_W3.txt'
		w4_path = prefix + '_W4.txt'
		b1_path = prefix + '_B1.txt'
		b2_path = prefix + '_B2.txt'
		b3_path = prefix + '_B3.txt'
		b4_path = prefix + '_B4.txt'

	in_features = None; inputs = None; outputs = None; init_weight = None;
	input_data = pd.read_csv(in_path,sep='\t',index_col=0)

	input_data = sampling(input_data)
	samples	= input_data.index
	#if 'NC' not in in_path:
	#	input_data = sampling(input_data)
	#else:
	#	input_data = case_sampling(input_data)

	if w1_path != None and w2_path != None and w3_path != None:
		w1_data = pd.read_csv(w1_path,sep='\t',index_col=0)
		w2_data = pd.read_csv(w2_path,sep='\t',index_col=0)
		w3_data = pd.read_csv(w3_path,sep='\t',index_col=0)
		in_features = w1_data.index
		L1_features = w1_data.columns
		L2_features = w2_data.columns
		L3_features = w3_data.columns
		if 'weights_he' in w1_path: w_dist = 'he';
		if 'weights_xv' in w1_path: w_dist = 'xv';
	elif w_path != None:
		if 'ensemble' in w_path:
			w1_data = pd.read_csv(w_path+'_W1.txt',sep='\t',index_col=0)
			w2_data = pd.read_csv(w_path+'_W2.txt',sep='\t',index_col=0)
			w3_data = pd.read_csv(w_path+'_W3.txt',sep='\t',index_col=0)
		else:
			w1_data = pd.read_csv(w_path+'_prior_1_W1.txt',sep='\t',index_col=0)
			w2_data = pd.read_csv(w_path+'_prior_1_W2.txt',sep='\t',index_col=0)
			w3_data = pd.read_csv(w_path+'_prior_1_W3.txt',sep='\t',index_col=0)
		w_dist = 'xv'
	else:
		print('===default weights selected===')
		w1_data = pd.read_csv('weights_file/newPFC_geno_iso_weights_xv.txt',sep='\t',index_col=0)
		w2_data = pd.read_csv('weights_file/newPFC_iso_marker_weights_xv.txt',sep='\t',index_col=0)
		w3_data = pd.read_csv('weights_file/newPFC_marker_pheno_weights_xv_cell.txt',sep='\t',index_col=0)
		w_dist = 'xv'

	in_features = w1_data.index
	L1_features = w1_data.columns
	L2_features = w2_data.columns
	L3_features = w3_data.columns
	feats = (in_features, L1_features, L2_features, L3_features)

	W1 = w1_data[L1_features].values
	W2 = w2_data[L2_features].values
	W3 = w3_data[L3_features].values

	X = input_data[in_features].values
	Y = input_data['Schizo'].values

	if w4_path != None:
		w4_data = pd.read_csv(w4_path,sep='\t',index_col=0)
		W4 = w4_data.values


	B1 = np.array([0,0,0]); B2 = np.array([0,0,0]); B3 = np.array([0,0,0]); B4 = np.array([0,0,0]);
	if b1_path!=None and b2_path != None and b3_path != None:
		b1_data = pd.read_csv(b1_path,sep='\t',index_col=0)
		b2_data = pd.read_csv(b2_path,sep='\t',index_col=0)
		b3_data = pd.read_csv(b3_path,sep='\t',index_col=0)
		B1 = b1_data.values.flatten()
		B2 = b2_data.values.flatten()
		B3 = b3_data.values.flatten()
	if b4_path!=None:
		b4_data = pd.read_csv(b4_path,sep='\t',index_col=0)
		B4 = b4_data.values.flatten()


	if y1_path!=None and y2_path != None and y3_path != None:
		Y1_data = pd.read_csv(y1_path,sep='\t',index_col=0)
		Y2_data = pd.read_csv(y2_path,sep='\t',index_col=0)
		Y3_data = pd.read_csv(y3_path,sep='\t',index_col=0)

		Y1 = Y1_data[L1_features].loc[samples].values
		Y2 = Y2_data[L2_features].loc[samples].values
		Y3 = Y3_data[L3_features].loc[samples].values
		Y4 = input_data['Schizo'].values


	#batches = [50, 100, 250, 500, 750, 1000]
	#epochs = [300, 400, 500, 750, 1000, 2000]#, 4000, 6000, 8000, 10000]
	#rates = [0.1, 0.3, 0.05, 0.01, 0.005, 0.001, 0.0005]

	if params != None:
		conn_thr = params[0]; b_size = int(params[1]); epo = int(params[2]); rate = float(params[3]);
	multi_M = Manager()


	# Model save mode
	if mode == 'save':
		#model_save(X, Y, X, Y, W1, W2, W3, B1, B2, B3, b_size, epo, rate, False, linear, w_dist, out_path+'_prior',(in_features, L1_features, L2_features, L3_features))
		log_file = open(out_path+'_log.txt','w')
		log_file.write('Batch size: '+str(b_size)+'\n')
		log_file.write('Learning rate: '+str(rate)+'\n')
		log_file.write('Epoch: '+str(epo)+'\n')
		log_file.write('Command: '+str(' '.join(sys.argv))+'\n')
		log_file.close()

		for i in range(10):
			procs = []
			for j in range(1,11):
				num = 10*i + j
				proc = Process(target=model_save,args=(X, Y, X, Y, W1, W2, W3, B1, B2, B3, b_size, epo, rate, False, linear, w_dist,conn_thr, out_path+'_prior_'+str(num),feats,))
				procs.append(proc)
				proc.daemon = True
				proc.start()
			for proc in procs:
				proc.join()
			for proc in procs:
				if proc.is_alive():
					proc.terminate()

		for i in range(10):
			procs = []
			for j in range(1,11):
				num = 10*i + j
				proc = Process(target=model_save,args=(X, Y, X, Y, W1, W2, W3, B1, B2, B3, b_size, epo, rate, True, linear, w_dist,conn_thr, out_path+'_rand_'+str(num),feats,))
				procs.append(proc)
				proc.daemon = True
				proc.start()
			for proc in procs:
				proc.join()
			for proc in procs:
				if proc.is_alive():
					proc.terminate()

	# Cross validation mode for performance evaluation
	if mode == 'CV_eval':
		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

		log_file = open(out_path.split('.')[0]+'_log.txt','w')
		log_file.wite('connectivity threshold: '+str(conn_thr)+'\n')
		log_file.write('Batch size: '+str(b_size)+'\n')
		log_file.write('Learning rate: '+str(rate)+'\n')
		log_file.write('Epoch: '+str(epo)+'\n')
		log_file.write('Command: '+str(' '.join(sys.argv))+'\n')
		log_file.close()

		out_file = open(out_path,'w')
		out_raw_file = open(out_path.split('.')[0]+'_raw.txt','w')
		header = "Experiment\taccuracy\tprecision\trecall\n"
		header_raw = "Experiment\taccuracy\tTP\tTN\tFP\tFN\tAUROC\tAUPR\n"
		out_file.write(header)
		out_raw_file.write(header_raw)
		accs = [[],[],[]]
		Ps = [[],[],[]]
		Rs = [[],[],[]]
		cv_sets = []
		print(conn_thr, b_size, rate, epo)
		for tr, te in kfold.split(X, Y):
			cv_sets.append((tr,te))

		for i in range(1,101):
			print('SLEM_'+str(i))
			procs = []
			cvscores = multi_M.dict(); cvscores['acc']=list(); cvscores['P']=list();cvscores['R']=list(); cvscores['raw']=list();
			for tr, te in cv_sets:
				proc = Process(target=model_test,args=(X[tr],Y[tr], X[te], Y[te], W1, W2, W3, B1, B2, B3, b_size, epo, rate, 2,  linear, w_dist, conn_thr, cvscores,))
				procs.append(proc)
				proc.daemon = True
				proc.start()
			for proc in procs:
				proc.join()
			for proc in procs:
				if proc.is_alive():
					proc.terminate()
			mean_score = np.mean(np.array(cvscores['acc']))
			mean_P = np.mean(np.array(cvscores['P']))
			mean_R = np.mean(np.array(cvscores['R']))
			for j in range(len(cvscores['raw'])):
				raw_txt = 'SLEM_'+str(i)+'-'+str(j)+'\t'+cvscores['raw'][j].replace(',','\t')+'\n'
				out_raw_file.write(raw_txt)

			#print(mean_score)
			accs[0].append(mean_score)
			Ps[0].append(mean_P)
			Rs[0].append(mean_R)
			txt = 'Prior_' +str(i)+ '\t' + str(mean_score) + '\t' + str(mean_P)+'\t'+str(mean_R)+'\n'
			out_file.write(txt)
			out_file.flush()
			out_raw_file.flush()
		for i in range(1,101):
			print('rand_'+str(i))
			procs = []
			cvscores = multi_M.dict(); cvscores['acc']=list(); cvscores['P']=list();cvscores['R']=list(); cvscores['raw']=list();
			for tr, te in cv_sets:
				proc = Process(target=model_test,args=(X[tr],Y[tr], X[te], Y[te], W1, W2, W3, B1, B2, B3, b_size, epo, rate, True,  linear, w_dist, conn_thr, cvscores,))
				procs.append(proc)
				proc.daemon = True
				proc.start()
			for proc in procs:
				proc.join()
			for proc in procs:
				if proc.is_alive():
					proc.terminate()

			mean_score = np.mean(np.array(cvscores['acc']))
			mean_P = np.mean(np.array(cvscores['P']))
			mean_R = np.mean(np.array(cvscores['R']))
			accs[1].append(mean_score)
			Ps[1].append(mean_P)
			Rs[1].append(mean_R)
			for j in range(len(cvscores['raw'])):
				raw_txt = 'random_'+str(i)+'-'+str(j)+'\t'+cvscores['raw'][j].replace(',','\t')+'\n'
				out_raw_file.write(raw_txt)

			txt = 'Random_' +str(i)+ '\t' + str(mean_score) + '\t' + str(mean_P)+'\t'+str(mean_R)+'\n'

			out_file.write(txt)
			out_file.flush()
			out_raw_file.flush()

		txt1 = 'Prior_mean' + '\t' + str(np.mean(np.array(accs[0]))) + '\t' + str(np.mean(np.array(Ps[0]))) + '\t' + str(np.mean(np.array(Rs[0]))) + '\n'
		txt2 = 'Random_mean' + '\t' + str(np.mean(np.array(accs[1]))) + '\t' + str(np.mean(np.array(Ps[1]))) + '\t' + str(np.mean(np.array(Rs[1]))) + '\n'
		out_file.write(txt1)
		out_file.write(txt2)

	# Hyperparameter tuning mode. Used for finding optimal parameter.
	if mode == 'tuning':
		batches = [32,64, 128, 256, 512, 1024]#, 3000]
		epochs = [200, 500, 750, 1000, 2000, 3000]#, 4000]#, 6000, 8000, 10000]
		rates = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
		conn_thrs = ['0.15','0.2','0.25','0.3','0.35','0.4']

		out_file = open(out_path,'w')
		header = "connection_thr\tbatches\tlearning_rate\tepoch\taccuracy\n"
		out_file.write(header)


		kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
		cv_sets = []
		for tr, te in kfold.split(X, Y):
			cv_sets.append((tr,te))
		for conn_thr in conn_thrs:
			prev_acc = 0.0; pprev_acc = 0.0;
			for b_size in batches:
				prev_acc = 0.0; pprev_acc = 0.0;
				for rate in rates:
					for epo in epochs:
						#b_size = batches[b]; rate = rates[b];
						print(conn_thr, b_size, epo, rate)
						exp_accs = []
						for exp_num in range(3):
							cvscores = multi_M.dict(); cvscores['acc']=list(); cvscores['P']=list();cvscores['R']=list(); cvscores['raw']=list();
							procs = []
							for tr, te in cv_sets:
								proc = Process(target=model_test,args=(X[tr],Y[tr], X[te], Y[te], W1, W2, W3, B1, B2, B3, b_size, epo, rate, rand,  linear, w_dist,conn_thr, cvscores,))
								procs.append(proc)
								proc.daemon = True
								proc.start()
							for proc in procs:
								proc.join()
							for proc in procs:
								if proc.is_alive():
									proc.terminate()

							exp_score = np.mean(np.array(cvscores['acc']))
							exp_accs.append(exp_score)
						mean_score = np.mean(np.array(exp_accs))

						txt = str(conn_thr)+'\t'+str(b_size) + '\t' + str(rate) + '\t' + str(epo) + '\t' + str(mean_score) + '\n'
						out_file.write(txt); out_file.flush()
						if pprev_acc > mean_score:
						    break;
						pprev_acc = prev_acc
						prev_acc = mean_score

if __name__ == '__main__':
	main()
