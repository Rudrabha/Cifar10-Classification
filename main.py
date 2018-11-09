import os
import numpy as np
from util import *
from sklearn import svm
from sklearn import tree

import argparse

from keras.models import Sequential
from  keras.layers  import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2

parser = argparse.ArgumentParser()
parser.add_argument("--create_dim_red", help="To create new dimension reduced data.", default=None)
parser.add_argument("--representation", help="Which represenatation to use", default="LDA", choices=["Raw_Data", "PCA", "LDA", "MLP_Embedding"])
parser.add_argument("--classifier", help="Which classifier to use, DT=Decision Tree, SVM, MLP, LR=Logistic Regression", default="DT", choices=["DT", "SVM", "LR", "MLP"])
args = parser.parse_args()


def select_representation(option):
	if option=="Raw_Data":
		trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels = load_cifar()
	elif option=="PCA":
		trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels = load_cifar()
		pca_train = joblib.load('Data/pca_train_40k_350.joblib') 
		val_data = transform_pca(val_data, pca_train)
		trn_data = transform_pca(trn_data, pca_train) 
		tst_data = transform_pca(tst_data, pca_train)

	elif option=="LDA":
		trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels = load_cifar()
		pca_train = joblib.load('Data/lda_40k.joblib') 
		val_data = transform_pca(val_data, pca_train)
		trn_data = transform_pca(trn_data, pca_train) 
		tst_data = transform_pca(tst_data, pca_train)
	elif option=="MLP_Embedding":
		_, trn_labels, val_data, val_labels, _, tst_labels = load_cifar()
		trn_data = np.load("Data/mlp_representation.npy")
		tst_data = np.load("Data/mlp_representation_tst.npy")
	return trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels

def dimensionality_reduction(option):
	if option=="PCA":
		trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels = load_cifar()
		pca_train, trn_data = find_pca(trn_data)
		joblib.dump(pca_train, 'Data/pca_train_40k_350.joblib')
	elif option=="LDA":
		trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels = load_cifar()
		lda_train = find_lda(trn_data, trn_labels)
		joblib.dump(lda_train, 'Data/lda_40k.joblib') 

def classifier(option, trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels):
	if option=="DT":
		svr = tree.DecisionTreeClassifier(max_depth=9)
		print ("Training Decision Tree")
		svr.fit(trn_data, trn_labels)
		print ("Predicting from model")
		pred = svr.predict(tst_data)
		acc, f1  = calculate_scores(tst_labels, pred)
		return acc, f1
	if option=="SVM":
		svr = svm.SVC(C=500.0, kernel="linear")
		print ("Training SVM")
		svr.fit(trn_data, trn_labels)
		print ("Predicting from model")
		pred = svr.predict(tst_data)
		acc, f1  = calculate_scores(tst_labels, pred)
		return acc, f1
	if option=="MLP":
		trn_labels, val_labels, tst_labels = convert_to_onehot(trn_labels, val_labels, tst_labels) 
		val_split = (tst_data, tst_labels)
		model = Sequential()
		layer1 = Dense(units = 1000, activation = 'relu', input_dim = trn_data.shape[1])
		model.add(layer1)
		model.add(Dropout(0.2))
		layer2 = Dense(units = 500, activation = 'relu')
		model.add(layer2)
		model.add(Dropout(0.2))
		layer3 = Dense(units = 250, activation = 'relu')
		model.add(layer3)
		model.add(Dropout(0.2))
		layer4 = Dense(units = 125, activation = 'relu')
		model.add(layer4)
		model.add(Dropout(0.2))
		layer5 = Dense(units = 50, activation = 'relu')
		model.add(layer5)
		model.add(Dropout(0.2))
		layer6 = Dense(units = 10, activation = 'softmax')
		model.add(layer6)
		optimizer = Adam(lr=1e-4)
		model.summary()
		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics = ['accuracy'])
		model.fit(trn_data, trn_labels, batch_size=5000, epochs=10, shuffle=True, validation_data=val_split)
		acc = model.evaluate(tst_data, tst_labels)
		return acc[1], acc[1]
	if option=="LR":
		trn_labels, val_labels, tst_labels = convert_to_onehot(trn_labels, val_labels, tst_labels) 
		val_split = (tst_data, tst_labels)
		model = Sequential()
		model.add(Dense(10, activation='softmax', input_dim=trn_data.shape[1]))
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
		model.fit(trn_data, trn_labels, batch_size=5000, epochs=10, shuffle=True, validation_data=val_split)
		acc = model.evaluate(tst_data, tst_labels)
		return acc[1], acc[1]

if args.create_dim_red=="PCA" or args.create_dim_red=="LDA":
	dimensionality_reduction(dim_red)
trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels = select_representation(args.representation)
acc, f1 = classifier(args.classifier, trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels)
print ("Accuracy = "+ str(acc))
print ("F1 Score = " + str(f1))

	
