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

    # Make an array for all the best fit params and the upper/lower limits.
    It_1_vals = []
    It_1_vals_up = []
    It_1_vals_low = []
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
            It_1_vals_up.append(It_1_up[j])
            It_1_vals_low.append(It_1_low[j])
            It_2_vals.append(It_2_val[j])
            It_2_vals_up.append(It_2_up[j])
            It_2_vals_low.append(It_2_low[j])

    Delta = (np.array(It_2_vals)- np.array(It_1_vals))

    Delta_1 = []
    Delta_err_low =[]
    Delta_err_up =[]
    for k, s in enumerate(Delta):
        if Delta[k] >= 0 :
            Delta_1.append(Delta[k]/It_1_vals_up[k])
            #Delta_err_low.append(It_2_vals_low[k]/It_1_vals_up[k])
            Delta_err_up.append(It_2_vals_up[k]/It_1_vals_up[k])
            Delta_err_low.append((np.array(It_2_vals[k])- np.array(It_1_vals[k]))/(It_1_vals_up[k]) + (It_2_vals_low[k] - np.array(It_2_vals[k]) +  np.array(It_1_vals[k]))/It_1_vals_low[k] )

        if Delta[k] < 0 :
            Delta_1.append(Delta[k]/It_1_vals_low[k])
            Delta_err_low.append(It_2_vals_low[k]/It_1_vals_low[k])
            #Delta_err_up.append(It_2_vals_up[k]/It_1_vals_low[k])
            Delta_err_up.append((abs(np.array(It_2_vals[k])- np.array(It_1_vals[k])))/(It_1_vals_low[k]) + (It_2_vals_up[k] - abs(np.array(It_2_vals[k]) -  np.array(It_1_vals[k])))/It_1_vals_up[k])

    #Delta_err_up = np.sqrt(np.array(It_2_vals_up)**2 + np.array(It_1_vals_up)**2)/np.array(It_1_vals_up)
    #Delta_err_low = np.sqrt(np.array(It_2_vals_low)**2 + np.array(It_1_vals_low)**2)/np.array(It_1_vals_low)

    #x = np.linspace(1, len(It_1_vals), len(It_1_vals))
    #fill = np.linspace(-10,5000,10)
    #ax = plt.axes()
    #plt.errorbar(x, Delta_1, yerr=(Delta_err_low, Delta_err_up), fmt="o", ecolor = 'k', color = 'k')
    #plt.ylim(-2,2)
    #plt.xlim(0,len(It_1_vals)+1)
    #ax.set_yticklabels(['','','-1','','0','','1','',''])
    #ax.set_xticklabels(['','','','','','',''])
    #ax.fill_between(fill, -1, 1, facecolor='grey', alpha=0.2)
    #ax.tick_params(which = 'both', bottom=False, top=False, left=False, right=False)
    #plt.hlines(0, -5, 500, color = 'k', linestyle= ':')


    y = np.linspace(1,len(It_1_vals), len(It_1_vals))
    fill = np.linspace(-1,1,10)
    ax = plt.axes()
    y[len(It_1_vals)-1] = y[len(It_1_vals)-1]+0.5
    plt.errorbar(Delta_1, y, xerr=(Delta_err_low, Delta_err_up), fmt="o", ecolor = 'k', color = 'k', markersize=3, elinewidth=1)
    plt.ylim(0,len(It_1_vals)+2)

    plt.xlim(-2,2)
    ax.set_xticklabels(['','','-1','','0','','1','',''])
    ax.set_yticklabels(['','','','','','',''])
    ax.fill_between(fill, -5, 200, facecolor='grey', alpha=0.2)
    ax.tick_params(which = 'both', bottom=False, top=False, left=False, right=False)
    plt.vlines(0, -5, 500, color = 'k', linestyle= ':')
    plt.vlines(0.5, -5, 500, color = 'k', linestyle= ':')
    plt.vlines(-0.5, -5, 500, color = 'k', linestyle= ':')

    ## Plotting models text & shading
    x = np.linspace(-10,10,10)


    ax.fill_between(x, 42.5, 45, facecolor='b', alpha=0.08)
    ax.text(-1.9,43.5,r'Flat $\Lambda$', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 40.5, 42.5, facecolor='r', alpha=0.08)
    ax.text(-1.9,41.5,r'$\Lambda$', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 38.5, 40.5, facecolor='b', alpha=0.08)
    ax.text(-1.9,39.5,r'F$w$CDM', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 35.5, 38.5, facecolor='r', alpha=0.08)
    ax.text(-1.9,37,r'$w$CDM', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 32.5, 35.5, facecolor='b', alpha=0.08)
    ax.text(-1.9,34,r'$w_0 w_a$', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 29.5, 32.5, facecolor='r', alpha=0.08)
    ax.text(-1.9,31,r'$w_0 w_z$', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 27.5, 29.5, facecolor='b', alpha=0.08)
    ax.text(-1.9,28.5,r'SCG', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 25.5, 27.5, facecolor='r', alpha=0.08)
    ax.text(-1.9,26.5,r'FGCG', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center')

    ax.fill_between(x, 22.5, 25.5, facecolor='b', alpha=0.08)
    ax.text(-1.9,24,r'GCG', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 18.5, 22.5, facecolor='r', alpha=0.08)
    ax.text(-1.9,20.5,r'NGCG', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 15.5, 18.5, facecolor='b', alpha=0.08)
    ax.text(-1.9,17,r'MPC', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 12.5, 15.5, facecolor='r', alpha=0.08)
    ax.text(-1.9,14,r'IDE1', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 8.5, 12.5, facecolor='b', alpha=0.08)
    ax.text(-1.9,10.5,r'IDE2', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, 4.5, 8.5, facecolor='r', alpha=0.08)
    ax.text(-1.9,6.5,r'IDE3', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center')

    ax.fill_between(x, 2.5, 4.5, facecolor='b', alpha=0.08)
    ax.text(-1.9,3.5,r'DGP', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    ax.fill_between(x, -1, 2.5, facecolor='r', alpha=0.08)
    ax.text(-1.9,1,r'GAL', family='serif',color='black',rotation=0,fontsize=12,ha='left', va='center') 

    plt.show()