import random
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas.core.common as com

from pandas.core.index import Index
from pandas.tools import plotting
from pandas.tools.plotting import scatter_matrix
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import learning_curve, validation_curve

#plt.figure(figsize=(12, 14))

def overlay(data1, data2, column, bins):
  """
  Overlay signal (data1) and background (data2) histograms for feature 'column'
  Control binning with 'bins'
  """

  data1 = data1[column]
  data2 = data2[column]

  minbin = min(data1.min(),data2.min())
  maxbin = max(data1.max(),data2.max())
  bins = np.linspace(minbin,maxbin,bins)

  plt.figure(figsize=(9, 6))
  ax = plt.subplot(111)    
  ax.spines["top"].set_visible(False)    
  ax.spines["right"].set_visible(False)    
  plt.xlabel(column, fontsize=20)  
  plt.ylabel("Normalized Counts", fontsize=20)  

  plt.hist(data1,bins,alpha=0.5,label='Signal', normed=1, log=False)
  plt.hist(data2,bins,alpha=0.5,label='Background', normed=1, log=False)
  plt.legend(loc='upper right',prop={'size':25})
  plt.savefig('plots/overlay_'+column+'.eps', format='eps', dpi=1000)
  plt.clf()

""" ****************************************************************************************** """

def scatter_plot(data, columnX, columnY, color='y'):
  """
  data : dataframe
  columnX : feature name on the x-axis
  columnY : feature name on the y-axis
  color : feature name on the target .. to determine how signal-like/background-like a point is
  """
  plt.figure(figsize=(9, 6))
  plt.xlabel(columnX, fontsize=20)  
  plt.ylabel(columnY, fontsize=20)  
  plt.xlim(data[columnX].min(),data[columnX].max())
  plt.ylim(data[columnY].min(),data[columnY].max())

  axes = data.plot(kind='scatter', x=columnX, y=columnY, c=color, cmap='autumn')
  axes.spines["top"].set_visible(False)    
  axes.spines["right"].set_visible(False)   

  plt.savefig('plots/scatter_'+columnX+'_'+columnY+'.eps', format='eps', dpi=1000)
  plt.clf()

""" ****************************************************************************************** """

def box_plot(data, column, by='y') : 
  """
  data : dataframe
  column: string or sequence
          If passed, will be used to limit data to a subset of columns
  
  """
  axes = data.boxplot(by=by,
           column=column,
           return_type='axes')

  return axes

""" ****************************************************************************************** """

# remove the y column from the correlation matrix
# after using it to select background and signal
##correlations(df[bg].drop('y', 1))
##correlations(df[sig].drop('y', 1))
##
def correlations(data, **kwds):
  """Calculate pairwise correlation between features.
  
  Extra arguments are passed on to DataFrame.corr()
  """
  # simply call df.corr() to get a table of
  # correlation values if you do not need
  # the fancy plotting
  corrmat = data.corr(**kwds)

  plt.figure(figsize=(9, 6))
  fig, ax1 = plt.subplots(ncols=1, figsize=(6,5))
    
  opts = {'cmap': plt.get_cmap("RdBu"),
            'vmin': -1, 'vmax': +1}
  heatmap1 = ax1.pcolor(corrmat, **opts)
  plt.colorbar(heatmap1, ax=ax1)

  ax1.set_title("Correlations")

  labels = corrmat.columns.values
  for ax in (ax1,):
    # shift location of ticks to center of the bins
    ax.set_xticks(np.arange(len(labels))+0.5, minor=False)
    ax.set_yticks(np.arange(len(labels))+0.5, minor=False)
    ax.set_xticklabels(labels, minor=False, ha='right', rotation=70)
    ax.set_yticklabels(labels, minor=False)
        
  plt.tight_layout()
  plt.savefig('plots/correlations.eps', format='eps', dpi=1000)
  plt.clf()

""" ****************************************************************************************** """

# roc curve is like an efficiency plot [True positive vs. False positive]
def plot_roc_curve(proba, X_test, y_test) : 
  """
  classifier : e.g DecisionTreeClassifier
  X_test : ndarray of input test data
  y_test : tuple of true y results used for testing

  """
#  decisions = classifier.decision_function(X_test)
  decisions = proba
  # Compute ROC curve and area under the curve
  fpr, tpr, thresholds = roc_curve(y_test, decisions)
  roc_auc = auc(fpr, tpr)
  
  plt.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)'%(roc_auc))
  
  plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title('Charged Higgs signal and TTbar background')
  plt.legend(loc="lower right")
  plt.grid()

""" ****************************************************************************************** """

# roc curve is like an efficiency plot [True positive vs. False positive]
def plot_roc_curves(y_test, *args, **kwargs) : 
  plt.figure(figsize=(9, 6))    

  ax = plt.subplot(111)    
  ax.spines["top"].set_visible(False)    
  #ax.spines["bottom"].set_visible(False)    
  ax.spines["right"].set_visible(False)    
  #ax.spines["left"].set_visible(False) 
  ax.get_xaxis().tick_bottom()    
  ax.get_yaxis().tick_left()
  
  plt.title('Charged Higgs signal and TTbar background', fontsize=20)
  plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
  plt.xlim([-0.05, 1.05])
  plt.ylim([-0.05, 1.05])
  plt.xlabel('False Positive Rate', fontsize=20)
  plt.ylabel('True Positive Rate', fontsize=20)

  counter=0
  for key, decision in kwargs.iteritems():
    name = args[counter]
    print("Decision is {}".format(decision))
    print("name is {}".format(name))
    counter += 1
    fpr,tpr,thresholds = roc_curve(y_test,decision)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=1,label="{} (Area = {})".format(name, roc_auc))

  plt.legend(loc='lower right',prop={'size':15})
  plt.savefig('plots/roc_curves.eps', format='eps', dpi=1000)

def draw_learning_curve(clf, X, y):
  train_sizes, train_scores, test_scores = learning_curve(clf, X, y, verbose=2, n_jobs=4)

  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)


  plt.figure(figsize=(9, 6))    

  ax = plt.subplot(111)    
  ax.spines["top"].set_visible(False)    
  ax.spines["right"].set_visible(False)    

  ax.get_xaxis().tick_bottom()    
  ax.get_yaxis().tick_left()
  
  plt.title('Learning Curve', fontsize=20)
  plt.xlabel('Training samples', fontsize=20)
  plt.ylabel('Score', fontsize=20)
  plt.ylim([-0.05, 1.05])

  plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.2,
											                     color="r")
  plt.plot(train_sizes,train_scores_mean, 'o-',label='Training', color="r") 

  plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
		                      test_scores_mean + test_scores_std, alpha=0.2, color="b")

  plt.plot(train_sizes,test_scores_mean, 'o-',label='Testing', color="b") 

  plt.legend(loc='best',prop={'size':15})

def draw_validation_curve(clf, X, y, parameter, par_range):
  train_scores, test_scores = validation_curve(clf, X, y, parameter, par_range, verbose=2)

  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)

  plt.figure(figsize=(9, 6))    

  ax = plt.subplot(111)    
  ax.spines["top"].set_visible(False)    
  ax.spines["right"].set_visible(False)    

  ax.get_xaxis().tick_bottom()    
  ax.get_yaxis().tick_left()
  
  plt.title('Validation Curve', fontsize=20)
  plt.ylim([-0.05, 1.05])
  plt.xlabel('{} '.format(parameter), fontsize=20)
  plt.ylabel('Score', fontsize=20)

  plt.plot(par_range,train_scores_mean, lw=2, color="darkorange", label='Training') 

  plt.fill_between(par_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.05,
								                  color="darkorange", lw=2)

  plt.plot(par_range,test_scores_mean, lw=2, color="navy", label='Testing') 

  plt.fill_between(par_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.05,
								                  color="navy", lw=2)


  plt.legend(loc='best',prop={'size':15})
