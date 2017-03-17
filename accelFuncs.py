import numpy as np

def gen_block(A,blk_size):
	num_blk=int(np.floor(A.shape[0]/float(blk_size)))
	X=np.zeros((num_blk,blk_size))
	Y=np.zeros((num_blk,blk_size))
	Z=np.zeros((num_blk,blk_size))
	for i in range(num_blk):
		X[i,:]=A[i*blk_size:(i+1)*blk_size,0]
		Y[i,:]=A[i*blk_size:(i+1)*blk_size,1]
		Z[i,:]=A[i*blk_size:(i+1)*blk_size,2]

	return X,Y,Z


def comp_features(blocks):
	'''
  compute the mean,std, peak to peak, power, one lag correlation for each axis

	'''
	blk_size=blocks.shape[1]
	num_features=5
	features=np.zeros((blocks.shape[0],num_features))
	
	#mean
	features[:,0]=np.mean(blocks,1)
	#std
	features[:,1]=np.std(blocks,1)
	#peak to peak
	features[:,2]=np.max(blocks,1)-np.min(blocks,1)
	#power
	features[:,3]=np.sum(blocks**2,1)
	#one lag autocorrelation
	for i in range(blocks.shape[0]):
		features[i,4]=np.correlate(blocks[i,:]-features[i,0], blocks[i,:]-features[i,0], "full")[blk_size]/np.sum((blocks[i,:]-features[i,0])**2)

	return features


def gen_features(A,blk_size):
	blocksX,blocksY,blocksZ=gen_block(A,blk_size)
	featuresX=comp_features(blocksX)
	featuresY=comp_features(blocksY)
	featuresZ=comp_features(blocksZ)
	crossCorr=np.zeros((blocksX.shape[0],3))

	for i in range(blocksX.shape[0]):
		crossCorr[i,0]=np.correlate(blocksX[i,:]-np.mean(blocksX[i,:]),blocksY[i,:]-np.mean(blocksY[i,:]))#/(np.std(blocksX[i,:])*np.std(blocksY[i,:]))
		crossCorr[i,1]=np.correlate(blocksY[i,:]-np.mean(blocksY[i,:]),blocksZ[i,:]-np.mean(blocksZ[i,:]))#/(np.std(blocksY[i,:])*np.std(blocksZ[i,:]))
		crossCorr[i,2]=np.correlate(blocksX[i,:]-np.mean(blocksX[i,:]),blocksZ[i,:]-np.mean(blocksZ[i,:]))#/(np.std(blocksX[i,:])*np.std(blocksZ[i,:]))
	

	features=np.concatenate((featuresX.T,featuresY.T,featuresZ.T,crossCorr.T)).T

	return features
