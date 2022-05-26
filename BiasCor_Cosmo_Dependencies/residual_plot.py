# This file produces a risidual plot in mu. This risudal plot characterises the Bias in the underlying
# simulations used for the Bias Corrections.

import numpy as np
import sys
from Hzs import *
sys.path.append('Cobaya_Chains')
from matplotlib import pyplot as plt
from matplotlib import rcParams, rc

model = 'IDEB_2'

# Data using Reference cosmology
norm_data = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains/data/DES5YR_REAL_DIFFIMG_DATA.txt", names=True)
#norm_data = np.genfromtxt("Cobaya_Chains/BiasCor_Cosmo_Dependencies/PIPPIN_OUTPUTS/%s/%s_hubble_diagram_NORM.txt" % (model, model), names=True)
#norm_data = np.genfromtxt("Cobaya_Chains/BiasCor_Cosmo_Dependencies/PIPPIN_OUTPUTS/%s_3/%s_hubble_diagram_MOD.txt" % (model, model), names=True)
mu_norm = norm_data['MU']
zz_norm = norm_data['zCMB']
ids_norm = norm_data['IDSURVEY'] # Used to seperate data into different colours
mu_norm_des = []
zz_des = []
zz_lowz = []
mu_norm_lowz = []
for i, id in enumerate(ids_norm):
    if id == 10:
        mu_norm_des.append(mu_norm[i])
        zz_des.append(zz_norm[i])
    else:
        mu_norm_lowz.append(mu_norm[i])
        zz_lowz.append(zz_norm[i])
mu_norm_des=np.array(mu_norm_des)
zz_des=np.array(zz_des)
mu_norm_lowz=np.array(mu_norm_lowz)
zz_lowz=np.array(zz_lowz)
print(len(mu_norm_des))
print(len(mu_norm_lowz))
print(len(mu_norm))
exit()

# Data after reperforming bias corrections
mod_data = np.genfromtxt("Cobaya_Chains/BiasCor_Cosmo_Dependencies/PIPPIN_OUTPUTS/%s/%s_hubble_diagram_MOD.txt" % (model, model), names=True)
#mod_data = np.genfromtxt("Cobaya_Chains/BiasCor_Cosmo_Dependencies/PIPPIN_OUTPUTS/%s_3/%s_3_hubble_diagram_MOD.txt" % (model, model), names=True)
mu_mod = mod_data['MU']
zz_mod = mod_data['zCMB']
ids_mod = mod_data['IDSURVEY'] # Used to seperate data into different colours
mu_mod_des = []
mu_mod_lowz = []
for i, id in enumerate(ids_mod):
    if id == 10:
        mu_mod_des.append(mu_mod[i])
    else:
        mu_mod_lowz.append(mu_mod[i])
mu_mod_des=np.array(mu_mod_des)
mu_mod_lowz=np.array(mu_mod_lowz)

# Create a plot
plt.rc('font', family='serif')
rcParams['mathtext.fontset'] = 'dejavuserif'
fig, axs = plt.subplots(1, 1, sharex=True,figsize=(5,4))
axs.set_xlabel('Redshift, z', fontsize=20)
axs.set_ylabel(r'$\Delta \mu$', fontsize=20)
axs.text(0.01,0.07,r'IDE2', family='serif',color='black',rotation=0,fontsize=12,ha='left') 

axs.axhline(0,color = 'k', linewidth=1, linestyle = '--', label = r'nominal analysis')
axs.scatter(zz_norm, mu_mod-mu_norm, color = '#FF0000', s=3, marker='o', alpha=0.5)
#axs.scatter(zz_lowz, mu_mod_lowz-mu_norm_lowz, color = 'orange', s=6, label = r'low-z')
axs.set_xscale('log')
#plt.legend(loc='lower left', frameon=False, fontsize=12)
axs.set_ylim(-0.09,0.09)
axs.set_xticklabels(['','','0.01','0.1','1.0'])
axs.set_yticklabels(['','-0.08','','-0.04','','0.00','','0.04','', '0.08'])
plt.tight_layout()
plt.savefig('Cobaya_Chains/BiasCor_Cosmo_Dependencies/Hubble_Comparison_Plots/%s.pdf' % model)  
plt.show()


# Left      - 0.17
# Bottom    - 0.15
# right     - 0.97
# Top       - 0.97