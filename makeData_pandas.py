#!/Users/lastferemenga/anaconda2/bin/python

import ROOT
import subprocess, sys
import numpy as np
import pandas as pd
from root_numpy import rec2array, root2array, root2rec, tree2rec, tree2array

sys.path.append('include/')

from chargedHbranches import *

# grab signal trees, merge them and assign a name
filename_signal = 'data/341524_0.root'
file_signal= ROOT.TFile(filename_signal)
sigTree = file_signal.Get('NOMINAL')

#convert signal tree to numpy array
signal = tree2rec(sigTree, branches, 'met_et>100000.&&event_clean&&n_vx>=1&&bsm_tj_dirty_jet==0', start=0)


# grab background trees, merge them and assign a name
filename_bkg = 'data/410000_0.root'
file_bkg= ROOT.TFile(filename_bkg)
bkgTree = file_bkg.Get('NOMINAL')

#convert background tree to numpy array
bkg = tree2rec(bkgTree, branches, 'met_et>100000.&&event_clean&&n_vx>=1&&bsm_tj_dirty_jet==0', start=0)
branchnames = bkg.dtype.names

signal = rec2array(signal)
bkg = rec2array(bkg)

X = np.concatenate((signal,bkg))
y = np.concatenate((np.ones(signal.shape[0]), np.zeros(bkg.shape[0])))

df = pd.DataFrame(X, columns=branchnames)
df['y'] = y

store = pd.HDFStore('data_arrays/chargedHtest.h5')
#store = pd.HDFStore('data_arrays/chargedH.h5')
store['df'] = df 
