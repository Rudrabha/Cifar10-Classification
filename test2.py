import os
import cv2
import numpy as np
from util import *
pca_train = joblib.load('Data/pca_train_24k.joblib') 
trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels = load_cifar()
val_data = transform_pca(val_data, pca_train)
trn_data = transform_pca(trn_data, pca_train) 
lr = logistic_regression(trn_data, trn_labels)
pred = predict(lr, val_data)
acc, f1 = calculate_scores(pred, val_labels)
print ("Accuracy for logistic regression: " + str(acc))
print ("F1 Score for logistic regression: " + str(f1))

