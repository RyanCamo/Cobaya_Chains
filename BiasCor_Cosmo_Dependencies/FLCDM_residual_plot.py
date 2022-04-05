# This file produces a risidual plot in mu. This risudal plot characterises the Bias in the underlying
# simulations used for the Bias Corrections.

import numpy as np
import sys
from Hzs import *
sys.path.append('Cobaya_Chains')
from matplotlib import pyplot as plt
from matplotlib import rcParams, rc

model = FLCDM.__name__

# Data using Reference cosmology
norm_data = np.genfromtxt("Cobaya_Chains/BiasCor_Cosmo_Dependencies/PIPPIN_OUTPUTS/%s/%s_hubble_diagram_NORM.txt" % (model, model), names=True)
mu_norm = norm_data['MU']
zz_norm = norm_data['zCMB']

# Data after reperforming bias corrections
mod_data = np.genfromtxt("Cobaya_Chains/BiasCor_Cosmo_Dependencies/PIPPIN_OUTPUTS/%s/%s_hubble_diagram_MOD.txt" % (model, model), names=True)
mu_mod = mod_data['MU']
zz_mod = mod_data['zCMB']

# Create a plot

rcParams['mathtext.fontset'] = 'dejavuserif'
fig, axs = plt.subplots(1, 1, sharex=True,figsize=(8,4))
axs.set_xlabel('Redshift, z', fontsize=22)
axs.set_ylabel(r'$\Delta \mu$', fontsize=22)

axs.axhline(0,color = 'k', linewidth=1, linestyle = '--', label = r'$\omega_{\mathrm{ref}}=-1.0$')
axs.scatter(zz_norm, mu_mod-mu_norm, color = 'b', s=6, label = r'$\omega_{\mathrm{ref}}=-1.028$', alpha=0.5)
plt.legend(loc='lower left', frameon=True,fontsize=14)
#axs.set_ylim(-0.01,0.01)
#ax.set_xticklabels(['0.1','','0.2','','0.3','','0.4','','0.5'])
#axs.set_yticklabels(['-0.10','','-0.05','','0','','0.05','', '0.10'])
plt.show()