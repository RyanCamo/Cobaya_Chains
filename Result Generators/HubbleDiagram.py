from sqlite3 import paramstyle
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import numpy as np
from chainconsumer import ChainConsumer
from Models_DL import *
import sys
sys.path.append('Cobaya_Chains')
from model_info import *

# This function gets the best fit parameters for the specific chains used.
def get_param(samples, label, weights, burnBAO_CMB_SN):
    c = ChainConsumer()
    c.add_chain(samples[burnBAO_CMB_SN:], parameters=label, linewidth=2.0, name="MCMC", weights=weights[burnBAO_CMB_SN:], kde=1.5, color="red").configure(summary=True,shade_alpha=0.3,statistics="cumulative")
    params = []
    for i, labelx in enumerate(label):
        params.append(c.analysis.get_summary(chains="MCMC")[labelx][1])
    c.remove_chain("MCMC")
    return params

# This function loads in the chain and then gets the best fit params from get_params.
def get_bestparams(model,label):
    # Makes a note of which columns to use.
    cols = []
    for i, l in enumerate(label):
        cols.append(i+2)
    burnSN, burnBAO_CMB, burnBAO_CMB_SN  = np.loadtxt('Cobaya_Chains/Contours/OUTPUT/BURNIN/%s_Burnin.txt' % (model))
    main_chain = np.loadtxt('Cobaya_Chains/chains/CMB+BAO+SN/%s_CMB_BAO_SN.1.txt' %(model), usecols=(cols), comments='#')
    weights = np.loadtxt('Cobaya_Chains/chains/CMB+BAO+SN/%s_CMB_BAO_SN.1.txt' %(model), usecols=(0), comments='#')
    params_main = get_param(main_chain,label, weights, int(burnBAO_CMB_SN) )
    return params_main

bin_data = np.genfromtxt("Cobaya_Chains/data/BIN_DES5YR_LOWZ_DATA.txt", names=True)
mu_bin = bin_data['MU']
mu_bin_err = bin_data['MUERR']
z_bin = bin_data['zCMB']
norm_bias = np.genfromtxt("Cobaya_Chains/data/UNBIN_DES5YR_LOWZ_data.txt", names=True, comments='#')
mu_norm = norm_bias['MU']
mu_err = norm_bias['MUERR']
z_data = norm_bias['zCMB']
ids = norm_bias['IDSURVEY']
z_range = np.linspace(0.009,1.45,2000) # Array of redshifts to plot models on.
DES = []
zz_DES = []
err_DES = []
LOWZ = []
zz_LOWZ = []
err_LOWZ = []
# Splits up into DES and LOWZ samples
for i, id in enumerate(ids):
    if id == 10:
        DES.append(mu_norm[i])
        zz_DES.append(z_data[i])
        err_DES.append(mu_err[i])
    if (id != 10):
        LOWZ.append(mu_norm[i])
        zz_LOWZ.append(z_data[i])
        err_LOWZ.append(mu_err[i])

# Plot
from matplotlib import rcParams, rc
rcParams['mathtext.fontset'] = 'dejavuserif'
fig, axs = plt.subplots(1, 1, sharex=True,figsize=(12,7))

# Models to put on the Hubble Diagram
models = [FLCDM, LCDM, FwCDM, wCDM, Fwa, Fwz, Chap, FGChap, GChap, NGCG, FCa, IDEA, IDEB, IDEC, DGP, GAL]

#Empty Universe Scaling
mu_EMPTY_data = LCDM(np.array(z_data), 0, 0)
mu_EMPTY_dataDES = LCDM(np.array(zz_DES), 0, 0) 
mu_EMPTY_dataLOWZ = LCDM(np.array(zz_LOWZ), 0, 0) 
mu_EMPTY_range = LCDM(z_range, 0, 0) 
mu_EMPTY_binrange = LCDM(z_bin, 0,0) 

# Finding the best offset for the EMPTY model.  
data_EMPTY_offset = (np.sum(mu_norm-mu_EMPTY_data)/len(z_data))
bindata_EMPTY_offset = (np.sum(mu_bin-mu_EMPTY_binrange)/len(z_bin))
mu_EMPTY_range_norm = mu_EMPTY_range + data_EMPTY_offset

#Plotting the Data wrt to the EMPTY universe.
axs.errorbar(zz_DES,DES-mu_EMPTY_dataDES - data_EMPTY_offset,yerr=err_DES,fmt='.',elinewidth=1,markersize=2, color='b' , alpha=0.1 )
axs.errorbar(zz_LOWZ,LOWZ-mu_EMPTY_dataLOWZ - data_EMPTY_offset,yerr=err_LOWZ,fmt='.',elinewidth=1,markersize=2, color='orange', alpha=0.1 )
#axs.errorbar(zz_DES,DES-mu_EMPTY_dataDES - data_EMPTY_offset,yerr=err_DES,fmt='.',elinewidth=1,markersize=2, color='grey' , alpha=0.1 )
#axs.errorbar(zz_LOWZ,LOWZ-mu_EMPTY_dataLOWZ - data_EMPTY_offset,yerr=err_LOWZ,fmt='.',elinewidth=1,markersize=2, color='grey', alpha=0.1 )

# Info for legend
from cycler import cycle
models_overleaf = [r'Flat $\Lambda$', r'$\Lambda$', r'F$w$CDM', r'$w$CDM', r'$w_0 w_a$', r'$w_0 w_z$', 'SCG', 'FGCG', 'GCG', 'NGCG', 'MPC', 'IDE1', 'IDE2', 'IDE3','DGP','GAL']
colours = ['k', 'm', 'g', 'r', 'lime']
colourcycler = cycle(colours)

lines = ["--","-", "-.",":"]
linecycler = cycle(lines)

for i, model in enumerate(models):
    label, begin, legend = get_info(model.__name__)
    params = get_bestparams(model.__name__, label)
    print(params)
    mu_model_data = model(np.array(z_data), *params) # Best fit over the data (used for normalising)
    mu_model_range = model(z_range, *params) # Best fit over the data (used for plotting)
    data_model_offset = (np.sum(mu_norm-mu_model_data)/len(z_data))
    mu_model_range_norm = mu_model_range + data_model_offset
    axs.plot(z_range, mu_model_range_norm-mu_EMPTY_range_norm, linestyle = next(linecycler), color=next(colourcycler),label=models_overleaf[i])

# Plot binned data.
axs.errorbar(z_bin,mu_bin-mu_EMPTY_binrange-bindata_EMPTY_offset,yerr=mu_bin_err,fmt='.',elinewidth=2,markersize=10, color='k')


# Plot settings
axs.set_ylabel(r'$\Delta$ Distance Modulus', fontsize=18)
axs.set_xlabel(r'Redshift', fontsize=18)
axs.set_xlim(0.008,1.4)
plt.minorticks_on()
axs.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in", labelsize=14)
axs.set_ylim(-0.41,0.41)
axs.legend(loc='upper left',frameon=False, ncol=4, fontsize=14)
axs.set_xscale('log')
axs.set_xticklabels(['','', '0.01','0.1','1.0'])
plt.savefig("Result Generators/hubble_diagram.png",bbox_inches='tight')
plt.show()