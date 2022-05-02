import numpy as np
import sys
sys.path.append('Cobaya_Chains')
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt
from model_info import *
import scipy.special as sc



def get_param(samples, label, weights, burnSN):
    c = ChainConsumer()
    c.add_chain(samples[burnSN:], parameters=label, linewidth=2.0, name="MCMC", weights=weights[burnSN:], kde=1.5, color="red").configure(summary=True,shade_alpha=0.3,statistics="cumulative")
    params = []
    params_upper = []
    params_lower = []
    for i, labelx in enumerate(label):
        params_upper.append(c.analysis.get_summary(chains="MCMC")[labelx][2] - c.analysis.get_summary(chains="MCMC")[labelx][1])
        params_lower.append(c.analysis.get_summary(chains="MCMC")[labelx][1] - c.analysis.get_summary(chains="MCMC")[labelx][0])
        params.append(c.analysis.get_summary(chains="MCMC")[labelx][1])
    return params, params_upper, params_lower


# This function loads in the chain and then calls get_params to get the best fits.
def get_bestparams(model):
    # Makes a note of which columns to use.
    label, begin, legend = get_info(model)
    cols = []
    for i, l in enumerate(label):
        cols.append(i+2)

    burnSN, burnBAO_CMB, burnBAO_CMB_SN  = np.loadtxt('Cobaya_Chains/Contours/OUTPUT/BURNIN/%s_Burnin.txt' % (model))
    print(model)

    It_1 = np.loadtxt('Cobaya_Chains/chains/SN/%s_SN.1.txt' %(model), usecols=(cols), comments='#')
    It_1_weights = np.loadtxt('Cobaya_Chains/chains/SN/%s_SN.1.txt' %(model), usecols=(0), comments='#')
    It_2 =  np.loadtxt('Cobaya_Chains/chains/SN_BiasCor/%s_SN_2.1.txt' %(model), usecols=(cols), comments='#')
    It_2_weights =  np.loadtxt('Cobaya_Chains/chains/SN_BiasCor/%s_SN_2.1.txt' %(model), usecols=(0), comments='#') 

    It_1_val, It_1_up, It_1_low = get_param(It_1,label, It_1_weights, int(burnSN))
    It_2_val, It_2_up, It_2_low = get_param(It_2,label, It_2_weights, int(burnSN))

    return It_1_val, It_1_up, It_1_low, It_2_val, It_2_up, It_2_low, len(label)

if __name__ == "__main__":
    models = ['GAL','DGP', 'IDEC', 'IDEB', 'IDEA', 'FCa', 'NGCG', 'GChap', 'FGChap', 'Chap', 'Fwz', 'Fwa', 'wCDM', 'FwCDM', 'LCDM', 'FLCDM']
    #models = ['GAL']
    # Make an array for all the best fit params and the upper/lower limits.
    It_1_vals = []
    It_1_vals_avg = []
    It_2_vals = []
    It_2_vals_up = []
    It_2_vals_low = []
    params_tot = 0

    # Get the best fits/upper limits for each model and append them to the above array while keeping track 
    # of how many parameters we are storing - might not need this
    for i, model in enumerate(models):
        It_1_val, It_1_up, It_1_low, It_2_val, It_2_up, It_2_low, num = get_bestparams(model)
        for j, s in enumerate(It_1_val):
            It_1_vals.append(It_1_val[j])
            It_1_vals_avg.append((It_1_up[j]+It_1_low[j])/2)
            It_2_vals.append(It_2_val[j])
            It_2_vals_up.append(It_2_up[j])
            It_2_vals_low.append(It_2_low[j])
    print(len(It_1_vals))

    Delta = (np.array(It_2_vals)- np.array(It_1_vals))

    Delta_scaled = []
    Delta_err_up_scaled =[]
    Delta_err_low_scaled =[]
    for k, s in enumerate(Delta):
        Delta_scaled.append(Delta[k]/It_1_vals_avg[k])
        Delta_err_up_scaled.append(It_2_vals_up[k]/It_1_vals_avg[k])
        Delta_err_low_scaled.append(It_2_vals_low[k]/It_1_vals_avg[k])


    y = np.linspace(1,len(It_1_vals), len(It_1_vals))
    fill = np.linspace(-1,1,10)
    ax = plt.axes()
    y[len(It_1_vals)-1] = y[len(It_1_vals)-1]+0.5
    plt.errorbar(Delta_scaled, y, xerr=(Delta_err_low_scaled, Delta_err_up_scaled), fmt="o", ecolor = 'k', color = 'k', markersize=3, elinewidth=1, capsize=5)
    plt.ylim(0,len(It_1_vals)+2)

    plt.xlim(-3,3)
    ax.set_xticklabels(['','','$-1\overline{\sigma}$','0','$1\overline{\sigma}$','',''])
    ax.set_yticklabels(['','','','','','',''])
    ax.fill_between(fill, -5, 200, facecolor='grey', alpha=0.2)
    ax.tick_params(which = 'both', bottom=False, top=False, left=False, right=False)
    plt.vlines(0, -5, 500, color = 'k', linestyle= ':')
    #plt.vlines(0.5, -5, 500, color = 'k', linestyle= ':')
    #plt.vlines(-0.5, -5, 500, color = 'k', linestyle= ':')

    ## Plotting models text & shading
    x = np.linspace(-10,10,10)

    x_start = -2.9
    ax.fill_between(x, 42.5, 45, facecolor='b', alpha=0.08)
    ax.text(x_start,43.5,r'Flat $\Lambda$', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 40.5, 42.5, facecolor='r', alpha=0.08)
    ax.text(x_start,41.5,r'$\Lambda$', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 38.5, 40.5, facecolor='b', alpha=0.08)
    ax.text(x_start,39.5,r'F$w$CDM', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 35.5, 38.5, facecolor='r', alpha=0.08)
    ax.text(x_start,37,r'$w$CDM', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 32.5, 35.5, facecolor='b', alpha=0.08)
    ax.text(x_start,34,r'$w_0 w_a$', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 29.5, 32.5, facecolor='r', alpha=0.08)
    ax.text(x_start,31,r'$w_0 w_z$', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 27.5, 29.5, facecolor='b', alpha=0.08)
    ax.text(x_start,28.5,r'SCG', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 25.5, 27.5, facecolor='r', alpha=0.08)
    ax.text(x_start,26.5,r'FGCG', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center')

    ax.fill_between(x, 22.5, 25.5, facecolor='b', alpha=0.08)
    ax.text(x_start,24,r'GCG', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 18.5, 22.5, facecolor='r', alpha=0.08)
    ax.text(x_start,20.5,r'NGCG', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 15.5, 18.5, facecolor='b', alpha=0.08)
    ax.text(x_start,17,r'MPC', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 12.5, 15.5, facecolor='r', alpha=0.08)
    ax.text(x_start,14,r'IDE1', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 8.5, 12.5, facecolor='b', alpha=0.08)
    ax.text(x_start,10.5,r'IDE2', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 4.5, 8.5, facecolor='r', alpha=0.08)
    ax.text(x_start,6.5,r'IDE3', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center')

    ax.fill_between(x, 2.5, 4.5, facecolor='b', alpha=0.08)
    ax.text(x_start,3.5,r'DGP', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, -1, 2.5, facecolor='r', alpha=0.08)
    ax.text(x_start,1,r'GAL', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    plt.show()