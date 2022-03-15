import numpy as np
from scipy.integrate import quad
import sys
sys.path.append('Cobaya_Chains')
from model_info import *
from matplotlib import markers, pyplot as plt
from chainconsumer import ChainConsumer
from scipy.stats import norm 
import matplotlib.patches as mpatches

c = ChainConsumer() 

model = 'wCDM'

# Importing the relevant chains. The output format from Cobaya places the chains from column 2.

# Chains for the 3 contours we wish to plot
SN = np.loadtxt('Cobaya_Chains/chains/SN/DES5YR/UNBIN/%s_DES5YR_UNBIN2.1.txt' %(model), usecols=(2, 3, 4), comments='#')
BAO_CMB = np.loadtxt('Cobaya_Chains/chains/CMB+BAO/%s_CMB_BAO.1.txt' %(model), usecols=(2, 3, 4), comments='#')
BAO_CMB_SN = np.loadtxt('Cobaya_Chains/chains/CMB+BAO+SN/%s_CMB_BAO_SN2.1.txt' %(model), usecols=(2, 3, 4), comments='#')

# MANUALLY CHANGE THE BURN HERE.
burnSN = 320
burnBAO_CMB = 200
burnBAO_CMB_SN = 200
np.savetxt('Cobaya_Chains/Contours/OUTPUT/BURNIN/%s_Burnin.txt' % (model), [burnSN, burnBAO_CMB, burnBAO_CMB_SN], fmt="%10.0f")

# Get Info for the model 
label, begin, legend = get_info(model)
print(label) # print this so I can check which axis im plotting by default (label[0] & label[1])

## We also wish to overlay FLCDM onto the contour
## FLCDM Chains & Best fit parameters for BAO/CMB+SN:
#label_FLCDM, begin_FLCDM, legend_FLCDM = get_info('FLCDM')
#FLCDM_BAO_CMB_SN = np.loadtxt('Cobaya_Chains/chains/CMB+BAO+SN/FLCDM_CMB_BAO_SN.1.txt', usecols=(2), comments='#')
#c.add_chain(FLCDM_BAO_CMB_SN, parameters=label_FLCDM, linewidth=2.0, name="FLCDM_BAO_CMB_SN", kde=1.5, color="grey").configure(summary=True, shade_alpha=0.2,statistics="max")
#fom = c.analysis.get_summary(chains="FLCDM_BAO_CMB_SN")[r'$\Omega_m$'][1]
#fomp = c.analysis.get_summary(chains="FLCDM_BAO_CMB_SN")[r'$\Omega_m$'][2]-c.analysis.get_summary(chains="FLCDM_BAO_CMB_SN")[r'$\Omega_m$'][1]
#fomm = c.analysis.get_summary(chains="FLCDM_BAO_CMB_SN")[r'$\Omega_m$'][1]-c.analysis.get_summary(chains="FLCDM_BAO_CMB_SN")[r'$\Omega_m$'][0]
#c.remove_chain('FLCDM_BAO_CMB_SN')
## add the below line to the plotting section:
##ax.scatter(fom, 1-fom, marker = 'D', s = 50, c='black', label = r'Flat $\Lambda$')


# Adding the model chains to chainconsumer to plot & plotting things
fig, ax = plt.subplots(1, 1)
c.add_chain(SN[burnSN:], parameters=label, linewidth=1.0, name="SN", kde=1.5, color="red",num_free_params=len(begin))
c.add_chain(BAO_CMB[burnBAO_CMB:], parameters=label, linewidth=1.0, name="BAO/CMB",  kde=1.5, color="#FFD700",num_free_params=len(begin))
c.add_chain(BAO_CMB_SN[burnBAO_CMB_SN:], parameters=label, linewidth=1.0, name="BAO/CMB+SN",  kde=1.5, color="#1E90FF",num_free_params=len(begin))
c.configure(summary=True, shade_alpha=1,statistics="max")

# Best Fit Parameters
# SN Only
p0_sn = c.analysis.get_summary(chains="SN")[label[0]][1]
p0p_sn = c.analysis.get_summary(chains="SN")[label[0]][2]-c.analysis.get_summary(chains="SN")[label[0]][1]
p0m_sn = c.analysis.get_summary(chains="SN")[label[0]][1]-c.analysis.get_summary(chains="SN")[label[0]][0]
p1_sn = c.analysis.get_summary(chains="SN")[label[2]][1]
p1p_sn =c.analysis.get_summary(chains="SN")[label[2]][2]-c.analysis.get_summary(chains="SN")[label[2]][1]
p1m_sn =c.analysis.get_summary(chains="SN")[label[2]][1]-c.analysis.get_summary(chains="SN")[label[2]][0]
print('%s = %.5s^{%.5s}_{%.5s}$' %(label[0],p0_sn,p0p_sn,p0m_sn))
print('%s = %.5s^{%.5s}_{%.5s}$' %(label[2],p1_sn,p1p_sn,p1m_sn))
# BAO/CMB + SN
p0_tot = c.analysis.get_summary(chains="BAO/CMB+SN")[label[0]][1]
p0p_tot = c.analysis.get_summary(chains="BAO/CMB+SN")[label[0]][2]-c.analysis.get_summary(chains="BAO/CMB+SN")[label[0]][1]
p0m_tot = c.analysis.get_summary(chains="BAO/CMB+SN")[label[0]][1]-c.analysis.get_summary(chains="BAO/CMB+SN")[label[0]][0]
p1_tot = c.analysis.get_summary(chains="BAO/CMB+SN")[label[2]][1]
p1p_tot =c.analysis.get_summary(chains="BAO/CMB+SN")[label[2]][2]-c.analysis.get_summary(chains="BAO/CMB+SN")[label[2]][1]
p1m_tot =c.analysis.get_summary(chains="BAO/CMB+SN")[label[2]][1]-c.analysis.get_summary(chains="BAO/CMB+SN")[label[2]][0]
print('%s = %.5s^{%.5s}_{%.5s}$' %(label[0],p0_tot,p0p_tot,p0m_tot))
print('%s = %.5s^{%.5s}_{%.5s}$' %(label[2],p1_tot,p1p_tot,p1m_tot))
#c.remove_chain("BAO/CMB+SN")

#ax.text(0.55,0.57,'$\Omega_m = %10.5s\pm{%10.5s}$' %(om,omp), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
#ax.text(0.55,0.57-0.05,'$\Omega_{\Lambda} = %10.5s\pm{%10.5s}$' %(ol,olp), family='serif',color='black',rotation=0,fontsize=12,ha='right')

xaxis = label[0] # Which slice to plot?
yaxis = label[2] # Which slice to plot?
c.plotter.plot_contour(ax,xaxis, yaxis)
ax.set_xlabel(xaxis, fontsize = 18)
ax.set_ylabel(yaxis, fontsize = 18) 
ax.set_xlim(0,0.7)
ax.set_ylim(-2,0)
#ax.set_xticklabels(['0.1','','0.2','','0.3','','0.4','','0.5'])
#ax.set_yticklabels(['-2.0','','-1.5','','-1.0','-0.5','','0.0'])

plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")

red_patch = mpatches.Patch(color='#FF0000', label='SN', ec='k')
yellow_patch = mpatches.Patch(color='#FFD700', label='CMB/BAO', ec='k')
blue_patch = mpatches.Patch(color='#1E90FF', label='SN+CMB/BAO', ec='k')
ax.legend(handles=[red_patch, yellow_patch, blue_patch], loc='upper left',frameon=False,fontsize=16)
#ax.scatter(fom, -1, marker = 'D', s = 20, c='black', label = r'Flat $\Lambda$')
ax.scatter(p0_tot, p1_tot, marker = 'D', s = 20, c='black', label = r'Flat $\Lambda$')
plt.show()