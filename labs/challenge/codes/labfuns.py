from __future__ import absolute_import, division, print_function
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
from matplotlib.colors import ColorConverter
import random as rnd
from sklearn.datasets import make_blobs
#from sklearn.datasets.samples_generator import make_blobs
from sklearn import decomposition, tree

# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data.
def trteSplit(X,y,pcSplit,seed=None):
    # Compute split indices
    Ndata = X.shape[0]
    Ntr = int(np.rint(Ndata*pcSplit))
    Nte = Ndata-Ntr
    np.random.seed(seed)    
    idx = np.random.permutation(Ndata)
    trIdx = idx[:Ntr]
    teIdx = idx[Ntr:]
    # Split data
    xTr = X[trIdx,:]
    yTr = y[trIdx]
    xTe = X[teIdx,:]
    yTe = y[teIdx]
    return xTr,yTr,xTe,yTe,trIdx,teIdx


# Splits data into training and test set, pcSplit defines the percent of
# the data should be used as training data. The major difference to
# trteSplit is that we select the percent from each class individually.
# This means that we are assured to have enough points for each class.
def trteSplitEven(X,y,pcSplit,seed=None):
    labels = np.unique(y)
    xTr = np.zeros((0,X.shape[1]))
    xTe = np.zeros((0,X.shape[1]))
    yTe = np.zeros((0,),dtype=int)
    yTr = np.zeros((0,),dtype=int)
    trIdx = np.zeros((0,),dtype=int)
    teIdx = np.zeros((0,),dtype=int)
    np.random.seed(seed)
    for label in labels:
        classIdx = np.where(y==label)[0]
        NPerClass = len(classIdx)
        Ntr = int(np.rint(NPerClass*pcSplit))
        idx = np.random.permutation(NPerClass)
        trClIdx = classIdx[idx[:Ntr]]
        teClIdx = classIdx[idx[Ntr:]]
        trIdx = np.hstack((trIdx,trClIdx))
        teIdx = np.hstack((teIdx,teClIdx))
        # Split data
        xTr = np.vstack((xTr,X[trClIdx,:]))
        yTr = np.hstack((yTr,y[trClIdx]))
        xTe = np.vstack((xTe,X[teClIdx,:]))
        yTe = np.hstack((yTe,y[teClIdx]))

    return xTr,yTr,xTe,yTe,trIdx,teIdx


def fetchDataset(dataset='iris'):
    if dataset == 'iris':
        X = genfromtxt('irisX.txt', delimiter=',')
        y = genfromtxt('irisY.txt', delimiter=',',dtype=np.int64)-1
        pcadim = 2
    elif dataset == 'challenge':
       from sklearn.preprocessing import LabelEncoder
       df = pd.read_csv('TrainOnMe-1d3c80f1-d0b9-436e-9bcf-f50f94c66e0b.csv')
       y = df.iloc[:, 1].values  
       X = df.drop(df.columns[[0, 1, 13]], axis=1)
       encoder = LabelEncoder()
       X.iloc[:, 6] = encoder.fit_transform(X.iloc[:, 6])
       X = X.values
       encoder1 = LabelEncoder()
       y = encoder1.fit_transform(y)
       pcadim = 12       
    return X,y,pcadim


# The function below, `testClassifier`, will be used to try out the different datasets.
# `fetchDataset` can be provided with any of the dataset arguments `wine`, `iris`, `olivetti` and `vowel`.
# Observe that we split the data into a **training** and a **testing** set.
def testClassifier(classifier, dataset='iris', dim=0, split=0.7, ntrials=100):

    X,y,pcadim = fetchDataset(dataset)

    means = np.zeros(ntrials,);

    for trial in range(ntrials):

        xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X,y,split,trial)

        # Do PCA replace default value if user provides it
        if dim > 0:
            pcadim = dim

        if pcadim > 0:
            pca = decomposition.PCA(n_components=pcadim)
            pca.fit(xTr)
            xTr = pca.transform(xTr)
            xTe = pca.transform(xTe)

        # Train
        trained_classifier = classifier.trainClassifier(xTr, yTr)
        # Predict
        yPr = trained_classifier.classify(xTe)

        # Compute classification error
        if trial % 10 == 0:
            print("Trial:",trial,"Accuracy","%.3g" % (100*np.mean((yPr==yTe).astype(float))) )

        means[trial] = 100*np.mean((yPr==yTe).astype(float))

    print("Final mean classification accuracy ", "%.3g" % (np.mean(means)), "with standard deviation", "%.3g" % (np.std(means)))
    return trained_classifier

class DecisionTreeClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, Xtr, yTr, W=None):
        rtn = DecisionTreeClassifier()
        rtn.classifier = tree.DecisionTreeClassifier(max_depth= int( Xtr.shape[1]/2+1) )
        if W is None:
            rtn.classifier.fit(Xtr, yTr)
        else:
            rtn.classifier.fit(Xtr, yTr, sample_weight=W.flatten())
        rtn.trained = True
        return rtn

    def classify(self, X):
        return self.classifier.predict(X)


def mlParams(X, labels, W=None):
    assert(X.shape[0]==labels.shape[0])
    Npts,Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    if W is None:
        W = np.ones((Npts,1))/float(Npts)

    mu = np.zeros((Nclasses,Ndims))
    sigma = np.zeros((Nclasses,Ndims,Ndims))
    # Iterate over both index and value
    for jdx,class_i in enumerate(classes):
        idx = np.where(labels==class_i)[0] # extract the indices for which y==class is true.
        xlc = X[idx,:]                     # Get the x for the class labels. Vectors are rows.
        mu[jdx,:] = np.mean(xlc, axis=0)   # axis=0: computes the mean of each feature (column).
        xlc_cen = xlc - mu[jdx,:]
        sigma[jdx,:,:] = np.dot(xlc_cen.T, xlc_cen) / xlc_cen.shape[0]  # .shape[0] means the row number.
        sigma[jdx, :, :] = np.diag(  np.diag(sigma[jdx, :, :])  ) # 1 extract the diagonal and 2 return as a matrix.    
    # ==========================
    return mu, sigma

def computePrior(labels, W=None):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts,1))/Npts
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)
    prior = np.zeros((Nclasses,1))
    for jdx,class_i in enumerate(classes):
        idx = np.where(labels==class_i)[0] # extract the indices for which y==class is true.        
        prior[jdx] = np.sum(W[idx])   

    return prior

def classifyBayes(X, prior, mu, sigma):
    Npts = X.shape[0]
    Nclasses,Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))
    for k in range(Nclasses):
        mu_k = mu[k,:]
        sigma_k = sigma[k,:,:]
        logDetSigma = np.sum( np.log(np.diag(sigma_k)))
        x_center = X - mu_k       
        inver_sigma = np.diag( 1/ np.diag(sigma_k) )
        item2 = np.sum( x_center @ inver_sigma * x_center, axis=1)
        logProb[k,:] = -0.5*logDetSigma - 0.5*item2 + np.log(prior[k,0])
    h = np.argmax(logProb,axis=0)
    return h

# NOTE: no need to touch this
class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W=None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)






#################################### boost related function ######################





# NOTE: no need to touch this
class BoostClassifier(object):
    def __init__(self, base_classifier, T=10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)


# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T=10):
    Npts,Ndims = np.shape(X)
    classifiers = [] # append new classifiers to this list
    alphas = [] # append the vote weight of the classifiers to this list
    wCur = np.ones((Npts,1))/float(Npts)  #step 0, weight is around 0.009
        
    for t in range(0, T):
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur)) 
        vote = classifiers[t].classify(X)  #vote is predicted y, is the label value, like 1, 2,3
        delta = []
        error = 0
        for i in range( len(vote) ):
            if vote[i] == labels[i].astype(float):
                delta.append(0)
            else:
                delta.append(1)         
        error = np.sum( wCur * ( np.array(delta).reshape(-1,1) ) )
        alpha = 0.5 * ( np.log(1- error) - np.log(error) )
        alphas.append(alpha) # you will need to append the new alpha

        w_new = [] # w_new is the new weights
        for i in range( len(wCur) ):
            if vote[i]==labels[i]:
                w_new.append(wCur[i]* np.exp(-alpha))
            else:
                w_new.append(wCur[i]* np.exp( alpha))
        zt = np.sum(w_new) # zt is the normalization factor, the sum of w_new.        
        wCur = np.array(w_new)/zt
        # ==========================
    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts,Nclasses))

        # TODO: implement classificiation when we have trained several classifiers!
        # here we can do it by filling in the votes vector with weighted votes
        # ==========================
        for t in range(Ncomps):
            yPred_t = classifiers[t].classify(X)            
            for i in range(Npts):
                votes[i, yPred_t[i]] += alphas[t]
        # ==========================
        # one way to compute yPred after accumulating the votes
        return np.argmax(votes,axis=1)