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

model = 'FCa'

# Get Info for the model 
label, begin, legend = get_info(model)
print(label) # print this so I can check which axis im plotting by default (label[0] & label[1])

cols = []
for i, l in enumerate(label):
    cols.append(i+2)

# Importing the relevant chains. The output format from Cobaya places the chains from column 2.

# Chains for the 3 contours we wish to plot
SN_2 = np.loadtxt('Cobaya_Chains/chains/SN_BiasCor/%s_SN_2.1.txt' %(model), usecols=(cols), comments='#')
SN_2_weights = np.loadtxt('Cobaya_Chains/chains/SN_BiasCor/%s_SN_2.1.txt' %(model), usecols=(0), comments='#')

SN_DES3YR = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/%s_DES3YR_UNBIN.1.txt' %(model), usecols=(cols), comments='#')
SN_DES3YR_weights = np.loadtxt('/Users/RyanCamo/Desktop/Cobaya/chains/%s_DES3YR_UNBIN.1.txt' %(model), usecols=(0), comments='#')
# google bbox_inches=tight


# Incase I want to plot FLCDM or LCDM
Flat_L_om = 0.315
L_om = 0.310
L_ol = 0.693


# Adding the model chains to chainconsumer to plot & plotting things
fig, ax = plt.subplots(1, 1)
c.add_chain(SN_DES3YR[200:], parameters=label, weights=SN_DES3YR_weights[200:],linewidth=1.0, name="DES3YR", kde=1.5, color="green",num_free_params=len(begin))
c.add_chain(SN_2, parameters=label, weights=SN_2_weights, linewidth=1.0, name="DES5YR", kde=1.5, color="red",num_free_params=len(begin))
c.configure(summary=True, shade_alpha=1, shade=[True, True],statistics="cumulative")

xaxis = label[1] # Which slice to plot?
yaxis = label[2] # Which slice to plot?
c.plotter.plot_contour(ax,xaxis, yaxis)
ax.set_xlabel(xaxis, fontsize = 18)
ax.set_ylabel(yaxis, fontsize = 18) 
ax.set_xlim(0.15,3)
ax.set_ylim(-3,0.66)
#ax.set_xlim(0,1)
#ax.set_ylim(0,1)
plt.minorticks_on()
ax.tick_params(which = 'both', bottom=True, top=True, left=True, right=True, direction="in", labelsize=14)


# Best Fit Parameters
# SN Only
p0_sn = c.analysis.get_summary(chains="DES5YR")[label[0]][1]
p0p_sn = c.analysis.get_summary(chains="DES5YR")[label[0]][2]-c.analysis.get_summary(chains="DES5YR")[label[0]][1]
p0m_sn = c.analysis.get_summary(chains="DES5YR")[label[0]][1]-c.analysis.get_summary(chains="DES5YR")[label[0]][0]
p1_sn = c.analysis.get_summary(chains="DES5YR")[label[1]][1]
p1p_sn =c.analysis.get_summary(chains="DES5YR")[label[1]][2]-c.analysis.get_summary(chains="DES5YR")[label[1]][1]
p1m_sn =c.analysis.get_summary(chains="DES5YR")[label[1]][1]-c.analysis.get_summary(chains="DES5YR")[label[1]][0]
print('%s = %.5s^{%.5s}_{%.5s}$' %(label[0],p0_sn,p0p_sn,p0m_sn))
print('%s = %.5s^{%.5s}_{%.5s}$' %(label[1],p1_sn,p1p_sn,p1m_sn))


#ax.text(0.55,0.57,'$\Omega_m = %10.5s\pm{%10.5s}$' %(om,omp), family='serif',color='black',rotation=0,fontsize=12,ha='right') 
#ax.text(0.55,0.57-0.05,'$\Omega_{\Lambda} = %10.5s\pm{%10.5s}$' %(ol,olp), family='serif',color='black',rotation=0,fontsize=12,ha='right')
#flatx = np.linspace(0,1,10)
#flaty = 1-flatx
#ax.plot(flatx, flaty, 'k--', alpha=0.9)
red_patch = mpatches.Patch(color='#FF0000', label='DES5YR', ec='k')
yellow_patch = mpatches.Patch(color='green', label='DES3YR', ec='k')
#blue_patch = mpatches.Patch(color='#1E90FF', label='SN+CMB/BAO', ec='k')
ax.legend(handles=[red_patch, yellow_patch], loc='lower right',frameon=False,fontsize=16)
ax.text(2.4,-2,r'MPC', family='serif',color='black',rotation=0,fontsize=16,ha='left') 
#ax.scatter(-1, 0, marker = 'o', s = 20, c='black', label = r'Flat $\Lambda$')
#ax.hlines(c.analysis.get_summary(chains="BAO/CMB+SN")[label[2]][2], -5, 5, colors='k', linestyles='--', alpha=0.9)
##ax.hlines(c.analysis.get_summary(chains="BAO/CMB+SN")[label[2]][0], -5, 5, colors='k', linestyles='--', alpha=0.9)
#ax.hlines(c.analysis.get_summary(chains="SN_2")[label[2]][2], -5, 5, colors='k', linestyles='--', alpha=0.9)
#ax.hlines(c.analysis.get_summary(chains="SN_2")[label[2]][0], -5, 5, colors='k', linestyles='--', alpha=0.9)
#plt.savefig('Cobaya_Chains/Contours/OUTPUT/%s.pdf' % (model), bbox_inches='tight', format = 'pdf')
plt.show()