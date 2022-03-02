import numpy as np
from scipy.integrate import quad
from model_info import *
from matplotlib import markers, pyplot as plt
from chainconsumer import ChainConsumer
from scipy.stats import norm 
import matplotlib.patches as mpatches

c = ChainConsumer() 

model = 'LCDM'

# Importing the relevant chains. The output format from Cobaya places the chains from column 2.

# Chains for the 3 contours we wish to plot
SN = np.loadtxt('Cobaya_Chains/chains/SN/DES5YR/UNBIN/%s_DES5YR_UNBIN.1.txt' %(model), usecols=(2, 3), comments='#')
BAO_CMB = np.loadtxt('Cobaya_Chains/chains/CMB+BAO/%s_CMB_BAO.1.txt' %(model), usecols=(2, 3), comments='#')
BAO_CMB_SN = np.loadtxt('Cobaya_Chains/chains/CMB+BAO+SN/%s_CMB_BAO_SN.1.txt' %(model), usecols=(2, 3), comments='#')

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
#ax.scatter(fom, 1-fom, marker = 'D', s = 50, c='black', label = r'Flat $\Lambda$')


# Adding the model chains to chainconsumer to plot & plotting things
fig, ax = plt.subplots(1, 1)
c.add_chain(SN, parameters=label, linewidth=1.0, name="SN", kde=1.5, color="red",num_free_params=len(begin))
c.add_chain(BAO_CMB[500:], parameters=label, linewidth=1.0, name="BAO/CMB", kde=1.5, color="#FFD700",num_free_params=len(begin))
c.add_chain(BAO_CMB_SN, parameters=label, linewidth=1.0, name="BAO/CMB+SN", color="#1E90FF",num_free_params=len(begin))
c.configure(summary=True, shade_alpha=1,statistics="max")

xaxis = label[0] # Which slice to plot?
yaxis = label[1] # Which slice to plot?
c.plotter.plot_contour(ax,xaxis, yaxis)
ax.set_xlabel(xaxis, fontsize = 18)
ax.set_ylabel(yaxis, fontsize = 18) 
ax.set_xlim(0.20,0.57)
ax.set_ylim(0.48,1.06)
#ax.set_xlim(0,1)
#ax.set_ylim(0,1.05)
plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")


# Best Fit Parameters
# SN Only
p0_sn = c.analysis.get_summary(chains="SN")[label[0]][1]
p0p_sn = c.analysis.get_summary(chains="SN")[label[0]][2]-c.analysis.get_summary(chains="SN")[label[0]][1]
p0m_sn = c.analysis.get_summary(chains="SN")[label[0]][1]-c.analysis.get_summary(chains="SN")[label[0]][0]
p1_sn = c.analysis.get_summary(chains="SN")[label[1]][1]
p1p_sn =c.analysis.get_summary(chains="SN")[label[1]][2]-c.analysis.get_summary(chains="SN")[label[1]][1]
p1m_sn =c.analysis.get_summary(chains="SN")[label[1]][1]-c.analysis.get_summary(chains="SN")[label[1]][0]
print('%s = %.5s^{%.5s}_{%.5s}$' %(label[0],p0_sn,p0p_sn,p0m_sn))
print('%s = %.5s^{%.5s}_{%.5s}$' %(label[1],p1_sn,p1p_sn,p1m_sn))
# BAO/CMB + SN
p0_tot = c.analysis.get_summary(chains="BAO/CMB+SN")[label[0]][1]
p0p_tot = c.analysis.get_summary(chains="BAO/CMB+SN")[label[0]][2]-c.analysis.get_summary(chains="BAO/CMB+SN")[label[0]][1]
p0m_tot = c.analysis.get_summary(chains="BAO/CMB+SN")[label[0]][1]-c.analysis.get_summary(chains="BAO/CMB+SN")[label[0]][0]
p1_tot = c.analysis.get_summary(chains="BAO/CMB+SN")[label[1]][1]
p1p_tot =c.analysis.get_summary(chains="BAO/CMB+SN")[label[1]][2]-c.analysis.get_summary(chains="BAO/CMB+SN")[label[1]][1]
p1m_tot =c.analysis.get_summary(chains="BAO/CMB+SN")[label[1]][1]-c.analysis.get_summary(chains="BAO/CMB+SN")[label[1]][0]
print('%s = %.5s^{%.5s}_{%.5s}$' %(label[0],p0_tot,p0p_tot,p0m_tot))
print('%s = %.5s^{%.5s}_{%.5s}$' %(label[1],p1_tot,p1p_tot,p1m_tot))


#ax.text(0.55,0.57,'$\Omega_m = %10.5s\pm{%10.5s}$' %(om,omp), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
#ax.text(0.55,0.57-0.05,'$\Omega_{\Lambda} = %10.5s\pm{%10.5s}$' %(ol,olp), family='serif',color='black',rotation=0,fontsize=12,ha='right')

red_patch = mpatches.Patch(color='#FF0000', label='SN')
yellow_patch = mpatches.Patch(color='#FFD700', label='CMB/BAO')
blue_patch = mpatches.Patch(color='#1E90FF', label='SN+CMB/BAO')
ax.legend(handles=[red_patch, yellow_patch, blue_patch], loc='upper left',frameon=False,fontsize=12)
plt.show()
