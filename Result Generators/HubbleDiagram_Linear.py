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

# Tam's Hubble diagram suggestion - explaining the BiasCor changes.

# Hubble Diagram after using H(z) = 50z + H_0 for BiasCor Sims
MOD_data = np.genfromtxt("Cobaya_Chains/BiasCor_Cosmo_Dependencies/PIPPIN_OUTPUTS/Fun/Fun_hubble_diagram_MOD.txt", names=True)
mu_MOD = MOD_data['MU']
mu_MOD_err = MOD_data['MUERR']
z_MOD = MOD_data['zCMB']

# Hubble Diagram  using Default sims for BiasCor Sims
norm_bias = np.genfromtxt("Cobaya_Chains/BiasCor_Cosmo_Dependencies/PIPPIN_OUTPUTS/Fun/Fun_hubble_diagram_NORM.txt", names=True, comments='#')
mu_norm = norm_bias['MU']
mu_err = norm_bias['MUERR']
z_data = norm_bias['zCMB']

z_range = np.linspace(0.009,1.45,2000) # Array of redshifts to plot models on.


# Plot
from matplotlib import rcParams, rc
rcParams['mathtext.fontset'] = 'dejavuserif'
fig, axs = plt.subplots(1, 1, sharex=True,figsize=(12,7))

plt.errorbar(z_data, mu_norm, markersize=2, color='b' , alpha=1,fmt='.' )
#plt.errorbar(z_MOD, mu_MOD,markersize=2, color='orange' , alpha=1,fmt='.' )

mu_Linear = Linear(z_range)
mu_Linear_data = Linear(z_data)
Linear_offset = (np.sum(mu_norm-mu_Linear_data)/len(z_data))
mu_FLCDM = FLCDM(z_range, 0.315)
mu_FLCDM_data = FLCDM(z_data, 0.315)
FLCDM_offset = (np.sum(mu_norm-mu_FLCDM_data)/len(z_data))
plt.plot(z_range, mu_Linear+Linear_offset, color = 'k', linestyle = "--")
plt.plot(z_range, mu_FLCDM+FLCDM_offset, color = 'k', linestyle = ":")



# Info for legend
from cycler import cycle
colours = ['k', 'm', 'g', 'r', 'lime']
colourcycler = cycle(colours)

lines = ["--","-", "-.",":"]
linecycler = cycle(lines)


# Plot settings
axs.set_ylabel(r'Distance Modulus', fontsize=18)
axs.set_xlabel(r'Redshift', fontsize=18)
axs.set_xlim(0.008,1.4)
plt.minorticks_on()
axs.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in", labelsize=14)
#axs.set_ylim(-0.41,0.41)
axs.legend(loc='upper left',frameon=False, ncol=4, fontsize=14)
axs.set_xscale('log')
axs.set_xticklabels(['','', '0.01','0.1','1.0'])
#plt.savefig("Result Generators/hubble_diagram_Linear.png",bbox_inches='tight')
plt.show()