import scipy.stats as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys, getopt, os, random

def readInputs(prefix):
	w1_path = prefix + '_W1.txt'
	w2_path = prefix + '_W2.txt'
	w3_path = prefix + '_W3.txt'
	w4_path = prefix + '_W4.txt'

	w1_data = pd.read_csv(w1_path,sep='\t',index_col=0)
	w2_data = pd.read_csv(w2_path,sep='\t',index_col=0)
	w3_data = pd.read_csv(w3_path,sep='\t',index_col=0)
	w4_data = pd.read_csv(w4_path,sep='\t',index_col=0)

	W1 = w1_data.values
	W2 = w2_data.values
	W3 = w3_data.values
	W4 = w4_data.values

	L1_features = np.array(w1_data.index.astype(str))
	L2_features = np.array(w2_data.index.astype(str))
	L3_features = np.array(w3_data.index.astype(str))
	L4_features = np.array(w3_data.columns.astype(str))

	return ((W1, W2, W3, W4),(L1_features,L2_features,L3_features,L4_features))

def readAnno(prefix):
	L2_dict = {}; L3_dict = {}; L4_dict = {};

	l2_path = prefix + '_L2.txt'
	l3_path = prefix + '_L3.txt'
	l4_path = prefix + '_L4.txt'

	l2_data = open(l2_path)
	l3_data = open(l3_path)
	l4_data = open(l4_path)

	for line in l2_data:
		atts = line.strip().split('\t')
		L2_dict[atts[0]] = atts[2]

	for line in l3_data:
		atts = line.strip().split('\t')
		L3_dict[atts[0]] = atts[0]+'_'+atts[1]

	for line in l4_data:
		atts = line.strip().split('\t')
		L4_dict[atts[0]] = atts[0]+'_'+atts[1]

	return (L2_dict, L3_dict, L4_dict)

def leftNonzeros(weights):
	W1 = np.array(weights[0]);  W2 = np.array(weights[1]); W3 = np.array(weights[2]); W4 = np.array(weights[3]);
	
	W1 = W1.flatten()
	W2 = W2.flatten()
	W3 = W3.flatten()
	W4 = W4.flatten()

	W1 = W1[W1!=0.0]
	W2 = W2[W2!=0.0]
	W3 = W3[W3!=0.0]
	W4 = W4[W4!=0.0]

	return (W1, W2, W3, W4)

def weightNormalize(weights):
	W1 = np.array(weights[0]);  W2 = np.array(weights[1]); W3 = np.array(weights[2]); W4 = np.array(weights[3]);
	nz_weights = leftNonzeros(weights)

	W1_mean = nz_weights[0].mean(); W1_std = nz_weights[0].std();
	W2_mean = nz_weights[1].mean(); W2_std = nz_weights[1].std();
	W3_mean = nz_weights[2].mean(); W3_std = nz_weights[2].std();
	W4_mean = nz_weights[3].mean(); W4_std = nz_weights[3].std();
	#print(W1_mean, W1_std)

	W1_conn = np.array(weights[0]);  W2_conn = np.array(weights[1]); W3_conn = np.array(weights[2]); W4_conn = np.array(weights[3]);

	W1_conn[W1_conn!=0.0] = 1; W2_conn[W2_conn!=0.0] = 1; W3_conn[W3_conn!=0.0] = 1; W4_conn[W4_conn!=0.0] = 1;

	W1 = (W1-W1_mean)/W1_std; W2 = (W2-W2_mean)/W2_std;  W3 = (W3-W3_mean)/W3_std; #W4 = (W4-W4_mean)/W4_std; 

	W1 = W1 * W1_conn; W2 = W2 * W2_conn; W3 = W3 * W3_conn; W4 = W4 * W4_conn;

	return (W1, W2, W3, W4)

def weightAnalysis(weights,paths):
	W1, W2, W3, W4 = leftNonzeros(weights)

	scores = []
	for p in paths:
		scores.append(p.getScore())
	scores = np.array(scores)

	fig = plt.figure()
	ax1 = fig.add_subplot(5,1,1)
	ax2 = fig.add_subplot(5,1,2)
	ax3 = fig.add_subplot(5,1,3)
	ax4 = fig.add_subplot(5,1,4)
	ax5 = fig.add_subplot(5,1,5)
	fig.tight_layout()

	ax1.hist(W1, bins= 20)
	ax2.hist(W2, bins= 20)
	ax3.hist(W3, bins= 20)
	ax4.hist(W4, bins= 20)
	ax5.hist(scores, bins = 20)

	plt.savefig('weight_dist.png', dpi=200, format='png')
	#plt.show()

	return None

class path:
	def __init__(self, nodes, scores):
		self.nodes = nodes
		self.node_ids = nodes
		self.weights = scores
		self.score = abs(scores[0]) + abs(scores[1]) + abs(scores[2])# + abs(scores[3])
	def __str__(self):
		txt = self.nodes[0] + '\t' + str(self.weights[0]) + '\t' + self.nodes[1] + '\t' + str(self.weights[1]) + '\t' + self.nodes[2] + '\t' + str(self.weights[2]) + '\t' + self.nodes[3] + '\t' + str(self.weights[3]) + '\tSchizo\t' + str(self.score)
		return txt

	def __lt__(self, other):
		#return self.score < other.score
		if abs(self.weights[3]) < abs(other.weights[3]):
			return True
		else:
			return False
		"""
		elif abs(self.weights[2]) < abs(other.weights[2]):
			return True
		elif abs(self.weights[1]) < abs(other.weights[1]):
			return True
		elif abs(self.weights[0]) < abs(other.weights[0]):
			return True
		"""


	def getIDs(self):
		return self.node_ids

	def getScore(self):
		return self.weights

	def getWeights(self):
		return self.weights

	def annotate(self, anno):
		self.nodes = anno

def findAllPaths(weights, features):
	path_list = []
	W1, W2, W3, W4 = weights
	L1, L2, L3, L4 = features

	for i in range(W4.shape[0]):
		n4 = L4[i]
		w4 = W4[i][0]
		if w4 == 0: continue;
		for j in range(W3.shape[0]):
			n3 = L3[j]
			w3 = W3[j][i]

			if w3 == 0: continue;
			for k in range(W2.shape[0]):
				n2 = L2[k]
				w2 = W2[k][j]
				if w2 == 0: continue;

				for l in range(W1.shape[0]):
					n1 = L1[l]
					w1 = W1[l][k]
					if w1 == 0: 
						continue;
					else:
						p = path((n1,n2,n3,n4),(w1,w2,w3,w4))
						path_list.append(p)
	return path_list

def findFilteredPaths(weights, features):
	path_list = []
	W1, W2, W3, W4 = weights
	L1, L2, L3, L4 = features
	filter_thr = [0.75, 0.895, 0.84, 0] #30%
	#filter_thr = [1.50, 1.557, 1.7, 2.69]
	markers = ['1754', '1769', '2459', '2460', '2774', '2779', '2780', '3151', '3152', '3179', '3626', '3646', '672', '722', '724', '731', '735', '736',\
	 '739', '742', '745', '753', '765', '780']

	for i in range(W4.shape[0]):
		n4 = L4[i]
		w4 = W4[i][0]
		if w4 == 0 or abs(w4) < filter_thr[3]: continue;
		for j in range(W3.shape[0]):
			n3 = L3[j]
			if n3 not in markers: continue;
			w3 = W3[j][i]
			if w3 == 0 or abs(w3) < filter_thr[2]: continue;
			for k in range(W2.shape[0]):
				n2 = L2[k]
				w2 = W2[k][j]
				if w2 == 0 or abs(w2) < filter_thr[1]: continue;
				for l in range(W1.shape[0]):
					n1 = L1[l]
					w1 = W1[l][k]
					if w1 == 0 or abs(w1) < filter_thr[0]: 
						continue;
					else:
						p = path((n1,n2,n3,n4),(w1,w2,w3,w4))
						path_list.append(p)
	return path_list

def getAnnotation(prefix, path_list):
	L2_dict, L3_dict, L4_dict = readAnno(prefix)
	out_list = []

	for p in path_list:
		ids = p.getIDs()
		p.annotate((ids[0],L2_dict[ids[1]],L3_dict[ids[2]],L4_dict[ids[3]]))
		out_list.append(p)

	return out_list

def listWeights(prefix, mode, W, F1, F2=None):
	L2_dict, L3_dict, L4_dict = readAnno(prefix)

	w_list = [];
	for i in range(W.shape[0]):
		if mode==1:
			lt = F1[i]
		if mode==2:
			#lt = L2_dict[F1[i]]
			lt = F1[i]
		if mode==3:
			#lt = L3_dict[F1[i]]
			lt = F1[i]
		if type(F2) is np.ndarray:
			for j in range(W.shape[1]):
				if mode == 1:
					#rt = L2_dict[F2[j]]
					rt = F2[j]
				if mode == 2:
					#rt = L3_dict[F2[j]]
					rt = F2[j]
				if mode == 3:
					#rt = L4_dict[F2[j]]
					rt = F2[j]
				if W[i][j] != 0:
					w_list.append((lt+'-'+rt,W[i][j] , abs(W[i][j])))
		else:
			rt = 'Schizo'
			#lt = L4_dict[F1[i]]
			lt = F1[i]
			if W[i][0] != 0:
				w_list.append((lt+'-'+rt, W[i][0], abs(W[i][0])))
	return w_list


def getTopWeights(prefix, weights, features):
	W1, W2, W3, W4 = weights
	L1, L2, L3, L4 = features

	W1_list = listWeights(prefix, 1,W1, L1, L2)
	W2_list = listWeights(prefix, 2,W2, L2, L3)
	W3_list = listWeights(prefix, 3,W3, L3, L4)
	W4_list = listWeights(prefix, 4,W4, L4)

	W1_list = sorted(W1_list,key=lambda x: x[2], reverse = True)
	W2_list = sorted(W2_list,key=lambda x: x[2], reverse = True)
	W3_list = sorted(W3_list,key=lambda x: x[2], reverse = True)
	W4_list = sorted(W4_list,key=lambda x: x[2], reverse = True)

	return (W1_list, W2_list, W3_list, W4_list)

def getTop5Paths(paths):
	phenos = []; p_IDs = [];
	for p in paths:
		if p.getIDs()[3] in p_IDs: continue;
		phenos.append([p.getIDs()[3],p.getWeights()[3]])
		p_IDs.append(p.getIDs()[3])
	phenos = sorted(list(phenos),key=lambda x: abs(x[1]), reverse=True)
	t5p = [x[0] for x in phenos]

	t5paths = []
	for p in paths:
		if p.getIDs()[3] in t5p:
			t5paths.append(p)
	return t5paths

def main():
	opts, args = getopt.getopt(sys.argv[1::],"o:",["model=","anno="])

	m_path = None; a_path = None; out_path = None;

	for o, a in opts:
			if o == "--model": m_path = a;
			if o == "--anno": a_path = a;
			if o == '-o': out_path = a;

	weights, features = readInputs(m_path)
	norm_weights = weightNormalize(weights)

	topweights = getTopWeights(a_path, norm_weights,features)

	#paths = findAllPaths(norm_weights, features)
	paths = findFilteredPaths(norm_weights, features)
	paths = getTop5Paths(paths)
	paths = sorted(paths, reverse = True)

	weightAnalysis(norm_weights, paths)

	paths = getAnnotation(a_path, paths)
	out = open(out_path,'w')

	for p in paths:
		out.write(str(p) + '\n')

	out2 = open(out_path.split('.')[0]+'_L1.txt','w')
	out3 = open(out_path.split('.')[0]+'_L2.txt','w')
	out4 = open(out_path.split('.')[0]+'_L3.txt','w')
	out5 = open(out_path.split('.')[0]+'_L4.txt','w')

	for w in topweights[0]:
		out2.write(w[0]+'\t'+str(w[1])+'\n')
	for w in topweights[1]:
		out3.write(w[0]+'\t'+str(w[1])+'\n')
	for w in topweights[2]:
		out4.write(w[0]+'\t'+str(w[1])+'\n')
	for w in topweights[3]:
		out5.write(w[0]+'\t'+str(w[1])+'\n')

if __name__ == '__main__':
	main()