import matplotlib.pyplot as plt
import numpy as np
import accelFuncs
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import pairwise_distances,accuracy_score
from sklearn.cross_validation import KFold,train_test_split

np.random.seed(123)
file1=open('HumanAccel',"r")


files=[]
labels=[]

while file1:
	
	line1=file1.readline()
	s1=line1.split()
	
	if len(line1)==0:
		break

	files.append(s1[0])
	labels.append(int(s1[1]))


blk_size=64

data=np.array([])
seglabels=[]
j=0

neigh=KNeighborsClassifier(n_neighbors=10)
for i in files:
	A=np.genfromtxt(i)
	F=accelFuncs.gen_features(A,blk_size)

	data= np.vstack([data,F]) if data.size else F
	seglabels=np.concatenate((seglabels,np.ones(F.shape[0])*labels[j]))
	j+=1

num_trials=100
recall=np.array([])
precision=np.array([])
f1=np.array([])
for i in range(num_trials):
	X_train, X_test, y_train, y_test = train_test_split(data, seglabels, test_size=0.3)
	neigh.fit(X_train,y_train)
	
	res=metrics.recall_score(y_test,neigh.predict(X_test),average=None)
	recall= np.vstack([recall,res]) if recall.size else res
	res=metrics.precision_score(y_test,neigh.predict(X_test),average=None)
	precision=np.vstack([precision,res]) if precision.size else res
	res=metrics.f1_score(y_test,neigh.predict(X_test),average=None)
	f1=np.vstack([f1,res]) if f1.size else res



print 'Precision',np.mean(precision,0)
print 'Recall',np.mean(recall,0)
print 'F1',np.mean(f1,0)
