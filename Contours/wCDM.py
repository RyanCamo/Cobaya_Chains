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
SN = np.loadtxt('Cobaya_Chains/chains/SN/%s_SN.1.txt' %(model), usecols=(2, 3, 4), comments='#')
SN_weights = np.loadtxt('Cobaya_Chains/chains/SN/%s_SN.1.txt' %(model), usecols=(0), comments='#')
SN_2 = np.loadtxt('Cobaya_Chains/chains/SN_BiasCor/%s_SN_2.1.txt' %(model), usecols=(2, 3, 4), comments='#')
SN_2_weights = np.loadtxt('Cobaya_Chains/chains/SN_BiasCor/%s_SN_2.1.txt' %(model), usecols=(0), comments='#')
BAO_CMB = np.loadtxt('Cobaya_Chains/chains/CMB+BAO/%s_CMB_BAO.1.txt' %(model), usecols=(2, 3, 4), comments='#')
BAO_CMB_weights = np.loadtxt('Cobaya_Chains/chains/CMB+BAO/%s_CMB_BAO.1.txt' %(model), usecols=(0), comments='#')
BAO_CMB_SN = np.loadtxt('Cobaya_Chains/chains/CMB+BAO+SN/%s_CMB_BAO_SN.1.txt' %(model), usecols=(2, 3, 4), comments='#')
BAO_CMB_SN_weights = np.loadtxt('Cobaya_Chains/chains/CMB+BAO+SN/%s_CMB_BAO_SN.1.txt' %(model), usecols=(0), comments='#')

# Incase I want to plot FLCDM or LCDM
Flat_L_om = 0.315
L_om = 0.310
L_ol = 0.693

# MANUALLY CHANGE THE BURN HERE.
burnSN = 0 #int(0.01*len(SN))
burnBAO_CMB = 0 #int(0.01*len(BAO_CMB))
burnBAO_CMB_SN = 0 #int(0.001*len(BAO_CMB_SN))
np.savetxt('Cobaya_Chains/Contours/OUTPUT/BURNIN/%s_Burnin.txt' % (model), [burnSN, burnBAO_CMB, burnBAO_CMB_SN], fmt="%10.0f")


# Get Info for the model 
label, begin, legend = get_info(model)
print(label) # print this so I can check which axis im plotting by default (label[0] & label[1])


# Adding the model chains to chainconsumer to plot & plotting things
fig, ax = plt.subplots(1, 1)
c.add_chain(SN_2, parameters=label, weights=SN_2_weights, linewidth=1.0, name="SN_2", kde=1.5, color="red",num_free_params=len(begin))
c.add_chain(SN, parameters=label, weights=SN_weights, linewidth=1.2, name="SN", kde=1.5, color="black", linestyle = '-.',num_free_params=len(begin))
c.add_chain(BAO_CMB, parameters=label, weights=BAO_CMB_weights,linewidth=1.0, name="BAO/CMB", kde=1.5, color="#FFD700",num_free_params=len(begin))
c.add_chain(BAO_CMB_SN, parameters=label, weights=BAO_CMB_SN_weights,linewidth=1.0, name="BAO/CMB+SN", kde=1.5, color="#1E90FF",num_free_params=len(begin))
c.add_chain(BAO_CMB, parameters=label, weights=BAO_CMB_weights,linewidth=1.0, name="BAO/CMB_2", kde=1.5, color="#FFD700", linestyle = '--', num_free_params=len(begin))
c.configure(summary=True, shade_alpha=1, shade=[True, False, True, True, False],statistics="max")

xaxis = label[0] # Which slice to plot?
yaxis = label[2] # Which slice to plot?
c.plotter.plot_contour(ax,xaxis, yaxis)
ax.set_xlabel(xaxis, fontsize = 18)
ax.set_ylabel(yaxis, fontsize = 18) 
ax.set_xlim(-0.2,0.55)
ax.set_ylim(-1.7,-0.5)
#ax.set_xlim(0,1)
#ax.set_ylim(0,1)
plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in")


# Best Fit Parameters
# SN Only
p0_sn = c.analysis.get_summary(chains="SN_2")[xaxis][1]
p0p_sn = c.analysis.get_summary(chains="SN_2")[xaxis][2]-c.analysis.get_summary(chains="SN_2")[xaxis][1]
p0m_sn = c.analysis.get_summary(chains="SN_2")[xaxis][1]-c.analysis.get_summary(chains="SN_2")[xaxis][0]
p1_sn = c.analysis.get_summary(chains="SN_2")[yaxis][1]
p1p_sn =c.analysis.get_summary(chains="SN_2")[yaxis][2]-c.analysis.get_summary(chains="SN_2")[yaxis][1]
p1m_sn =c.analysis.get_summary(chains="SN_2")[yaxis][1]-c.analysis.get_summary(chains="SN_2")[yaxis][0]
print('%s = %.5s^{%.5s}_{%.5s}$' %(xaxis,p0_sn,p0p_sn,p0m_sn))
print('%s = %.5s^{%.5s}_{%.5s}$' %(yaxis,p1_sn,p1p_sn,p1m_sn))
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
ax.hlines(-1, -5, 5, colors='k', linestyles='--', alpha=0.9)
#ax.plot(0, -1, 'k--', alpha=0.9)
red_patch = mpatches.Patch(color='#FF0000', label='SN', ec='k')
yellow_patch = mpatches.Patch(color='#FFD700', label='CMB/BAO', ec='k')
blue_patch = mpatches.Patch(color='#1E90FF', label='SN+CMB/BAO', ec='k')
ax.legend(handles=[red_patch, yellow_patch, blue_patch], loc='lower left',frameon=False,fontsize=16)
ax.scatter(L_om, -1, marker = 's', s = 20, c='black', label = r'$\Lambda$')
#plt.savefig('Cobaya_Chains/Contours/OUTPUT/%s.pdf' % (model), bbox_inches='tight', format = 'pdf')
plt.show()