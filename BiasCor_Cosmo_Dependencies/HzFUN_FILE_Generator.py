import numpy as np
import sys
from Hzs import *
sys.path.append('/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains')
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt
from model_info import *
import scipy.special as sc
from chainconsumer import ChainConsumer

c = ChainConsumer() 


# This file was designed to be used to check cosmological dependencies in the biascorrections part of Pippin.
# These files are used as sim-input keys: 'HzFUN_FILE:' when creating a SIM using SNANA.

# This code creates numerous files for different specified models.
# The output format is 2 columns: z, H(z) with the first row being z=0.


# Import the data - specifically the redshift range to mimic.
DES5YR_UNBIN = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains/data/UNBIN_DES5YR_LOWZ_data.txt", names=True)
zs = DES5YR_UNBIN['zCMB']
z_min = 0
z_max = np.max(zs)
z = np.linspace(0,1.4,500)
#hz = np.linspace(60, 130, 500)

#np.savetxt('/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains/BiasCor_Cosmo_Dependencies/Files_2/Fun.txt', np.c_[z, hz], fmt="%10.4f")
#exit()


models = [IDEC_2] #[FLCDM, LCDM, FwCDM, wCDM, FGChap,Fwa, Fwz, DGP, GAL]
# to do: 

#[FLCDM, LCDM, FwCDM, wCDM, IDE1, IDE2, IDE4, FGChap, GChap, FCa, Fwa, Fwz, DGP, GAL, NGCG]

for i, model in enumerate(models):
    #label, params, legend = get_info(model.__name__)
    label, params, legend = get_info(model.__name__)
    cols = []
    for s, parm in enumerate(params):
        cols.append(2+s)
    cols = np.array(cols)
    cols  = cols.tolist()
    #SN_2 = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains/chains/SN_BiasCor/%s_SN_2.1.txt' %(model.__name__), usecols=cols, comments='#')
    #weights = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains/chains/SN_BiasCor/%s_SN_2.1.txt' %(model.__name__), usecols=(0), comments='#')
    SN = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains/chains/SN/%s_SN.1.txt' %(model.__name__), usecols=cols, comments='#')
    weights = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains/chains/SN/%s_SN.1.txt' %(model.__name__), usecols=0, comments='#')
    c.add_chain(SN, parameters=label, weights=weights, linewidth=1.0, name="SN", kde=1.5, color="red",num_free_params=len(params)).configure(summary=True,shade_alpha=0.3,statistics="cumulative")
    #c.add_chain(SN_2, parameters=label, linewidth=1.0, weights=weights, name="SN_2", kde=1.5, color="red",num_free_params=len(params))
    best_fit = []
    for t, best in enumerate(params):
        best = c.analysis.get_summary(chains="SN")[label[t]][1]
        best_fit.append(best)
    c.remove_chain('SN')
    print(best_fit)
    Hz = model(z, *best_fit)
    np.savetxt('/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains/BiasCor_Cosmo_Dependencies/Files_2/%s.txt' % (model.__name__), np.c_[z, 70*Hz], fmt="%10.4f")

