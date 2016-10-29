#!/usr/bin/env python

import sys

import numpy as np
import pandas as pd

# Import the nearest neighbor class
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation, preprocessing
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier,ExtraTreesClassifier, AdaBoostClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, confusion_matrix
import myPackages.makePlots as plotter
import matplotlib.pyplot as plt

np.seterr(divide='ignore', invalid='ignore')

#########
#get data
#########

input = pd.read_hdf('data_arrays/chargedHtest.h5', 'df')
input = shuffle(input) #mix signal and background up
input.dropna()
columnNames = np.array(input.columns.values)

X = np.array(input.drop('y',1)).astype(float)
y = np.array(input['y'])
y = y.ravel() #convert it to a 1d array

#scikit likes to treat large numbers as infinities sometimes; scale around the mean
X = preprocessing.scale(X)

# split X into train and test sets
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.4, random_state=0)

# use randomforests to rank each variable, so that we can select which variables to use
n_ft=15
rf = RandomForestClassifier()
trf = RFE(rf,n_features_to_select=n_ft,verbose=1)
Xt_train = trf.fit_transform(X_train,y_train)
print("Reduced number of features ... {}".format(Xt_train.shape[1]))

approvedIndices = trf.get_support()
approvedNames = np.compress(approvedIndices, columnNames, axis=0)

###################
# Explore the data  
####################

# I prefer working with dataframes when plotting!
frame = pd.DataFrame(data=Xt_train,columns=approvedNames)
frame['y'] = y_train

#for feature in approvedNames:
#  #plot histos of signal vs. background
#  print("Printing {} overlays".format(feature))
#  plotter.overlay(frame[frame.y>0.5], frame[frame.y<0.5],feature,20)
#
#for feature in approvedNames:
#  for feature2 in approvedNames:
#    #scatter plot of two variables
#    print("Printing {} - {} scatter".format(feature,feature2))
#    plotter.scatter_plot(frame,feature,feature2)



#################
# Now training classifiers
##################

Xt_test = trf.transform(X_test)

from sklearn import svm

clf = svm.SVC()
clf.fit(Xt_train,y_train)

y_predicted_svm = clf.predict(Xt_test)
proba_svm = clf.decision_function(Xt_test)

print("Classification report for SVM")
print("****************")
print classification_report(y_test, y_predicted_svm)

print("Confusion matrix for SVM")
print("****************")
print confusion_matrix(y_test,y_predicted_svm)

print("Area under ROC curve: {}".format(roc_auc_score(y_test,proba_svm)))


#Nearest neighbor classifier
kN = KNeighborsClassifier()
kN.fit(Xt_train,y_train)

Xt_test = trf.transform(X_test)
y_predicted = kN.predict(Xt_test)
y_proba = kN.predict_proba(Xt_test)[:,1]
print("kNN proba is ... {}".format(y_proba))

print("Classification report for kNN")
print("****************")
print classification_report(y_test, y_predicted)

print("Confusion matrix for kNN")
print("****************")
print confusion_matrix(y_test,y_predicted)

print("Area under ROC curve: {}".format(roc_auc_score(y_test,y_proba)))


###############
# Neural Net
###############

from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

seed=7
np.random.seed(seed)

#build and compile model
def create_model():
  dnn_model = Sequential()
  dnn_model.add(Dense(n_ft,input_dim=n_ft,init='normal',activation='relu')) #hidden layer same features as input 
  dnn_model.add(Dense(1,init='normal',activation='sigmoid'))
  dnn_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

  return dnn_model

kclf = KerasClassifier(build_fn=create_model,nb_epoch=3,batch_size=5,verbose=2) # parameters unoptimized

kclf.fit(Xt_train,y_train)
predict_dnn = kclf.predict(Xt_test)
proba_dnn = kclf.predict_proba(Xt_test)[:,1]

print("Classification report for dNN")
print("****************")
print classification_report(y_test, predict_dnn)

print("Confusion matrix for dNN")
print("****************")
print confusion_matrix(y_test,predict_dnn)

print("Area under ROC curve: {}".format(roc_auc_score(y_test,proba_dnn)))

plotter.plot_roc_curves(y_test, "kNN","SVM", "dNN", clf0=proba_dnn, clf1=proba_svm, clf2=y_proba)

###################
# Examine kNN more
###################

plotter.draw_learning_curve(kN, Xt_train, y_train)
plt.savefig('plots/kNN_learningCurve.eps', format='eps', dpi=1000)
plt.clf()

plotter.draw_validation_curve(kN, Xt_train, y_train, 'n_neighbors', np.arange(2,20))
plt.savefig('plots/kNN_validationCurve.eps', format='eps', dpi=1000)
plt.clf()

