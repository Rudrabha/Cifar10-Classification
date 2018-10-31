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
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import OneHotEncoder

from tqdm import *
from functools import wraps
from time import time as _timenow 
from sys import stderr

def load_cifar():
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
	sp = 30000*0.8
	sp = int(sp)
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

def transform_pca(X, pca):
	X = pca.transform(X)
	return X

def predict(model, test_data):
	pred = model.predict(test_data)
	return pred

def logistic_regression(X, y):
	lr = LogisticRegression(solver="newton-cg", multi_class="multinomial", max_iter=1000)
	lr.fit(X, y)
	return lr

def sgd_logistic(X, y):
	clf = SGDClassifier(loss='log', max_iter=10000)
	clf.fit(X,y)
	return clf

def calculate_scores(gt, pred):
	acc = accuracy_score(gt, pred)
	f = f1_score(gt, pred, average='micro')
	return acc, f

def convert_to_onehot(train_labels, val_labels, test_labels):
	enc = OneHotEncoder()
	train_labels = enc.fit_transform(train_labels.reshape(-1, 1))
	test_labels = enc.transform(test_labels.reshape(-1, 1))
	val_labels = enc.transform(val_labels.reshape(-1, 1))
	return train_labels, val_labels, test_labels
