import os
import cv2
import numpy as np
from util import *
pca_train = joblib.load('Data/pca_train_24k.joblib') 
trn_data, trn_labels, val_data, val_labels, tst_data, tst_labels = load_cifar()
val_data = transform_pca(val_data, pca_train)
trn_data = transform_pca(trn_data, pca_train) 
tst_data = transform_pca(tst_data, pca_train)
#trn_labels, val_labels, tst_labels = convert_to_onehot(trn_labels, val_labels, tst_labels) 
lr = sgd_logistic(trn_data, trn_labels)
pred = predict(lr, val_data)
acc, f1 = calculate_scores(pred, val_labels)
pred_test = predict(lr, tst_data)
acc_test, f1_test = calculate_scores(pred_test, tst_labels)
print ("Accuracy for logistic regression: " + str(acc_test))
print ("F1 Score for logistic regression: " + str(f1_test))

