from sqlite3 import paramstyle
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

#norm_bias = np.genfromtxt("/Users/RyanCamo/Downloads/PIPPIN_OUTPUT/New Mock Data/UNBIN_DES5YR_LOWZ_DATA.txt", names=True)
norm_bias = np.genfromtxt("Cobaya_Chains/data/hubble_diagram.txt", names=True, comments='#')
mu_norm = norm_bias['MU']
mu_err = norm_bias['MUERR']
zz_norm = norm_bias['zCMB']
ids = norm_bias['IDSURVEY']
zz_best = np.linspace(0.009,1.45,2000)
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
        zz_DES.append(zz_norm[i])
        err_DES.append(mu_err[i])
    if (id != 10):
        LOWZ.append(mu_norm[i])
        zz_LOWZ.append(zz_norm[i])
        err_LOWZ.append(mu_err[i])

# Models to put on the Hubble Diagram
models = [IDEB]

# Plot settings
from matplotlib import rcParams, rc
rcParams['mathtext.fontset'] = 'dejavuserif'
fig, axs = plt.subplots(2, 1, sharex=True,figsize=(12,7),gridspec_kw = {'height_ratios':[1, 1]})
fig.subplots_adjust(hspace=0)
axs[0].set_ylabel(r'Distance Modulus, $\mu$', fontsize=16)
axs[1].set_xlabel('Redshift, z', fontsize=18)
axs[1].set_ylabel(r'$\mu$ Residual', fontsize=18)
axs[0].set_xlim(0.008,1.5)
axs[1].set_xlim(0.008,1.5)
axs[1].set_xscale('log')
axs[0].set_xscale('log')
axs[1].tick_params(axis='both', labelsize=14)
axs[0].tick_params(axis='both', labelsize=14)
plt.legend(loc='lower left', ncol=1,frameon=False,fontsize=14)
axs[0].set_xticklabels(['','','0.01','0.1','1.0'])
axs[1].set_ylim(-0.7,0.7)
axs[0].set_ylim(30,48)

bestfit_FLCDM = FLCDM(np.array(zz_norm), 0.31454268118243)
bestfit_FLCDMDES = FLCDM(np.array(zz_DES), 0.31454268118243)
bestfit_FLCDMLOWZ = FLCDM(np.array(zz_LOWZ), 0.31454268118243)
model_FLCDM = FLCDM(zz_best, 0.31454268118243) +(np.sum(mu_norm-bestfit_FLCDM)/len(zz_norm))
axs[0].plot(zz_best, model_FLCDM, 'k--')

for i, model in enumerate(models):
    label, begin, legend = get_info(model.__name__)
    params = get_bestparams(model.__name__, label)
    print(params)
    bestfit = model(np.array(zz_norm), *params)
    mu_model = model(zz_best, *params)+(np.sum(mu_norm-bestfit)/len(zz_norm))
    axs[0].plot(zz_best, mu_model, 'g--')
    axs[1].plot(zz_best, mu_model-model_FLCDM, 'g--')


axs[0].errorbar(zz_DES,DES,yerr=err_DES,fmt='.',elinewidth=1,markersize=2, color='b' , alpha=0.3 )
axs[0].errorbar(zz_LOWZ,LOWZ,yerr=err_LOWZ,fmt='.',elinewidth=1,markersize=2, color='orange', alpha=0.3 )
axs[1].errorbar(zz_DES, DES-bestfit_FLCDMDES -(np.sum(mu_norm-bestfit)/len(zz_norm)), yerr=err_DES,fmt='.',elinewidth=1,markersize=2, color='b' , alpha=0.3 )
axs[1].errorbar(zz_LOWZ, LOWZ-bestfit_FLCDMLOWZ -(np.sum(mu_norm-bestfit)/len(zz_norm)), yerr=err_LOWZ,fmt='.',elinewidth=1,markersize=2, color='orange' , alpha=0.3 )
axs[1].plot(zz_best, model_FLCDM-model_FLCDM, 'k--')
#axs[1].set_yticklabels(['','','0.01','0.1','1.0'])
#plt.xlabel('Redshift, z', fontsize=20)
#plt.ylabel(r'$\Delta$ Distance Modulus (Mag)', fontsize=20)
#plt.savefig("hubble_diagram.png",bbox_inches='tight')
plt.show()