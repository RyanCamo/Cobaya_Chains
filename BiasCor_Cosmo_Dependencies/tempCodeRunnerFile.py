# This file produces a risidual plot in mu. This risudal plot characterises the Bias in the underlying simulation
# used for the Bias Corrections.

import numpy as np
import sys
from Hzs import *
sys.path.append('Cobaya_Chains')
from matplotlib import pyplot as plt


# Import the mu data to compare:
# Pippin output after Bias corrections with Flat w: Om = 0.3 w = -1.
#norm_bias = np.genfromtxt("/Users/RyanCamo/Downloads/PIPPIN_OUTPUT/RC_BIASCOMPARE2/hubble_diagram_NORM1.txt", names=True)
#mu_norm = norm_bias['MU']
#zz_norm = norm_bias['zCMB']

norm_bias_bin = np.genfromtxt("/Users/RyanCamo/Downloads/PIPPIN_OUTPUT/Rep_bin/hubble_diagram_NORM.txt", names=True)
mu_norm_bin = norm_bias_bin['MU']
zz_norm_bin = norm_bias_bin['zCMB']

norm_bias = np.genfromtxt("/Users/RyanCamo/Downloads/PIPPIN_OUTPUT/Replicate/hubble_diagram_NORM.txt", names=True)
mu_norm = norm_bias['MU']
zz_norm = norm_bias['zCMB']


# Pippin output after Bias corrections with Om = 0.5 Ol = 0.5.
mod_bias_bin = np.genfromtxt("/Users/RyanCamo/Downloads/PIPPIN_OUTPUT/Rep_bin/hubble_diagram_MOD.txt", names=True)
mu_mod_bin = mod_bias_bin['MU']
zz_mod_bin = mod_bias_bin['zCMB']
mod_bias = np.genfromtxt("/Users/RyanCamo/Downloads/PIPPIN_OUTPUT/Replicate/hubble_diagram_Rep.txt", names=True)
mu_mod = mod_bias['MU']
zz_mod = mod_bias['zCMB']

x = np.sum(mu_mod_bin-mu_norm_bin)/len(zz_mod_bin)
#print(x)
#exit()

from matplotlib import rcParams, rc
rcParams['mathtext.fontset'] = 'dejavuserif'
fig, axs = plt.subplots(1, 1, sharex=True,figsize=(8,4))
fig.subplots_adjust(hspace=0)
axs.set_xlabel('Redshift, z', fontsize=22)
axs.set_ylabel(r'$\Delta \mu$', fontsize=22)
#axs[0].set_ylabel(r'Distance Modulus, $\mu$', fontsize=22)
axs.axhline(0,color = 'k', linewidth=1, linestyle = '--', label = r'$\omega_{\mathrm{ref}}=-1.0$')
axs.scatter(zz_norm, mu_mod-mu_norm, color = 'b', s=6, label = r'UNBIN: $\omega_{\mathrm{ref}}=-1.028$', alpha=0.5)
axs.scatter(zz_norm_bin, mu_mod_bin-mu_norm_bin, color = 'k', s=50, label = r'BIN: $\omega_{\mathrm{ref}}=-1.028$')
plt.legend(loc='lower right', frameon=True,fontsize=14)
#axs.set_ylim(-0.01,0.01)
#ax.set_xticklabels(['0.1','','0.2','','0.3','','0.4','','0.5'])
#axs.set_yticklabels(['-0.10','','-0.05','','0','','0.05','', '0.10'])
plt.show()