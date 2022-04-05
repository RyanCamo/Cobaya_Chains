import numpy as np
import sys
from Hzs import *
sys.path.append('Cobaya_Chains')
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
DES5YR_UNBIN = np.genfromtxt("data/DES5YR_UNBIN_data.txt", names=True)
zs = DES5YR_UNBIN['zCMB']
z_min = 0
z_max = np.max(zs)
z = np.linspace(0,1.4,500)


models = [FLCDM, LCDM, FwCDM, wCDM, IDE1, IDE2, IDE4, FGChap, GChap, FCa, Fwa, Fwz, DGP, GAL, NGCG]
#[FLCDM, LCDM, FwCDM, wCDM, IDE1, IDE2, IDE4, FGChap, GChap, FCa, Fwa, Fwz, DGP, GAL, NGCG]

for i, model in enumerate(models):
    label, params, legend = get_info(model.__name__)
    cols = []
    for s, parm in enumerate(params):
        cols.append(2+s)
    cols = np.array(cols)
    cols  = cols.tolist()
    SN = np.loadtxt('Cobaya_Chains/chains/SN/DES5YR/UNBIN/%s_DES5YR_UNBIN.1.txt' %(model.__name__), usecols=cols, comments='#')
    c.add_chain(SN, parameters=label, linewidth=1.0, name="SN", kde=1.5, color="red",num_free_params=len(params))
    best_fit = []
    for t, best in enumerate(params):
        best = c.analysis.get_summary(chains="SN")[label[t]][1]
        best_fit.append(best)
    c.remove_chain('SN')
    Hz = model(z, *best_fit)
    np.savetxt('Cobaya_Chains/BiasCor_Cosmo_Dependencies/Files/%s.txt' % (model.__name__), np.c_[z, 70*Hz], fmt="%10.4f")

