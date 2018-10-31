import os
import cv2
import pickle
import numpy as np
import pdb
import requests
from collections import defaultdict
import random 
import time

from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import *

from functools import wraps
from time import time as _timenow 
from sys import stderr

def load_cifar(sp=20000):
	trn_data, trn_labels, tst_data, tst_labels = [], [], [], []
	def unpickle(file):
		with open(file, 'rb') as fo:
			data = pickle.load(fo, encoding='latin1')
		return data
	for i in trange(3):
		batchName = 'cifar10/data_batch_{0}'.format(i + 1)
		unpickled = unpickle(batchName)
		trn_data.extend(unpickled['data'])
		trn_labels.extend(unpickled['labels'])
	unpickled = unpickle('cifar10/test_batch')
	tst_data.extend(unpickled['data'])
	tst_labels.extend(unpickled['labels'])
	trn_data = np.asarray(trn_data)
	trn_labels = np.asarray(trn_labels)
	tst_data = np.asarray(tst_data)
	tst_labels = np.asarray(tst_labels)
	val_data = trn_data[sp:].copy()
	val_labels = trn_labels[sp:].copy()
	trn_data = trn_data[0:sp]
	trn_labels = trn_labels[0:sp]
	return trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels


def find_pca(X, n=10):
	pca_train = IncrementalPCA(n_components=n, batch_size=10)
	pca_train.fit(X)
	X = pca_train.transform(X)
	return pca_train, X

def transform(X, pca):
	X = pca.transform(X)
	return X

trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels = load_cifar()
pca_train, trn_data = find_pca(trn_data)
print (trn_data.shape)
