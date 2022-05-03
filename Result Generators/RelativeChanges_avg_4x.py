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
    #models = ['wCDM']
    models = ['GAL','DGP', 'IDEC', 'IDEB', 'IDEA', 'FCa', 'NGCG', 'GChap', 'FGChap', 'Chap', 'Fwz', 'Fwa', 'wCDM', 'FwCDM', 'LCDM', 'FLCDM']

    fig, axs = plt.subplots(1, 4, figsize=(9, 3), sharey=True)

    # Get the best fits/upper limits for each model and append them to the above array while keeping track 
    # of how many parameters we are storing - might not need this
    dicts={}
    for i, model in enumerate(models):
        It_1_val, It_1_up, It_1_low, It_2_val, It_2_up, It_2_low, num = get_bestparams(model)
        dicts[model] = num
        #runnin_count = np.array(list(dicts.values()))
        #cumsum = runnin_count.cumsum(axis=1)
        for j, s in enumerate(It_1_val):
            delta = It_2_val[j] - It_1_val[j]
            avg_sig = (abs(It_1_up[j])+abs(It_1_low[j]))/2
            delta_scaled = delta/avg_sig
            delta_err_up_scaled = np.array(It_2_up[j]/avg_sig)
            delta_err_low_scaled = np.array(It_2_low[j]/avg_sig)
            if model == 'NGCG': # Just switching up the order (Om was not first in the chain)
                if j==0:
                    axs[j+1].errorbar(delta_scaled, i+1, xerr=np.array([[delta_err_low_scaled ,delta_err_up_scaled]]).T, fmt="o", ecolor = 'k', color = 'k', markersize=3, elinewidth=1, capsize=2)
                elif j==1:
                    axs[j+1].errorbar(delta_scaled, i+1, xerr=np.array([[delta_err_low_scaled ,delta_err_up_scaled]]).T, fmt="o", ecolor = 'k', color = 'k', markersize=3, elinewidth=1, capsize=2)
                elif j==2:
                    axs[0].errorbar(delta_scaled, i+1, xerr=np.array([[delta_err_low_scaled ,delta_err_up_scaled]]).T, fmt="o", ecolor = 'k', color = 'k', markersize=3, elinewidth=1, capsize=2)
                else:
                    axs[j].errorbar(delta_scaled, i+1, xerr=np.array([[delta_err_low_scaled ,delta_err_up_scaled]]).T, fmt="o", ecolor = 'k', color = 'k', markersize=3, elinewidth=1, capsize=2)                   
            elif model =='DGP': # DGP has no Om so pushed the parameter down by 1 for the plot
                axs[j+1].errorbar(delta_scaled, i+1, xerr=np.array([[delta_err_low_scaled ,delta_err_up_scaled]]).T, fmt="o", ecolor = 'k', color = 'k', markersize=3, elinewidth=1, capsize=2)
            #elif model =='IDE2': # Adding Ocdm and Ob for the first parameter to be Om
            #    if j==0:
            #        
            #        axs[j].errorbar(delta_scaled, i+1, xerr=np.array([[delta_err_low_scaled ,delta_err_up_scaled]]).T, fmt="o", ecolor = 'k', color = 'k', markersize=3, elinewidth=1, capsize=2)
            #    else:
            #        axs[j].errorbar(delta_scaled, i+1, xerr=np.array([[delta_err_low_scaled ,delta_err_up_scaled]]).T, fmt="o", ecolor = 'k', color = 'k', markersize=3, elinewidth=1, capsize=2)  
            else:
                axs[j].errorbar(delta_scaled, i+1, xerr=np.array([[delta_err_low_scaled ,delta_err_up_scaled]]).T, fmt="o", ecolor = 'k', color = 'k', markersize=3, elinewidth=1, capsize=2)


    # Things to apply to all subplots.
    Sub_Names = [r"$\Omega_{\mathrm{m}}$", 'Parameter 2', 'Parameter 3', 'Parameter 4']
    xinterval = np.linspace(-2,2,5) 
    yinterval = np.linspace(0,len(models)+1, len(models)+2)
    fill = np.linspace(-1,1,10) # Used for shading
    models_overleaf = ['GAL','DGP', 'IDE3', 'IDE2', 'IDE1', 'MPC', 'NGCG', 'GCG', 'FGCG', 'SCG', r'$w_0 w_z$', r'$w_0 w_a$', r'$w$CDM', r'F$w$CDM', r'$\Lambda$', r'Flat $\Lambda$']
    for i in range(4):
        axs[i].set_ylim(0,len(models)+1)
        axs[i].set_xlim(-2,2)
        axs[i].vlines(0, -5, 500, color = 'k', linestyle= ':')
        axs[i].fill_between(fill, -5, 200, facecolor='grey', alpha=0.2)
        axs[i].set_xticks(xinterval)
        axs[i].set_xticklabels(['','$-1\overline{\sigma}$','0','$1\overline{\sigma}$',''])
        axs[i].set_yticks(yinterval)
        axs[i].set_yticklabels(['', *models_overleaf, ''])
        axs[i].tick_params(which = 'both', bottom=False, top=False, left=False, right=False)
        y_min, y_max = axs[i].get_ylim()
        axs[i].text(0, 1.05*y_max, Sub_Names[i], va='center', ha='center')
    axs[0].text(-2,-2,r'Note for an Arb. estimate, $\mathrm{P}_{1}=x_{-~\sigma^{-}}^{+~\sigma^{+}}$, $\overline{\sigma}$ is defined as $\overline{\sigma}\equiv \frac{\sigma^{+}+\sigma^{-}}{2}$', family='serif',color='black',rotation=0,fontsize=10,ha='left', va='center')
    #axs[3].text(1.8,1.5,r'For an Arb. estimate:', family='serif',color='black',rotation=0,fontsize=10,ha='center', va='center')
    #axs[3].text(1.8,1.25,r'$\mathrm{P}_{1}=x_{-~\sigma^{-}}^{+~\sigma^{+}}$', family='serif',color='black',rotation=0,fontsize=10,ha='center', va='center')
    #axs[3].text(1.8,1,r'$\overline{\sigma}=\frac{\sigma^{+}+\sigma^{-}}{2}$', family='serif',color='black',rotation=0,fontsize=10,ha='center', va='center')
    
    #plt.text()
    #plt.vlines(0.5, -5, 500, color = 'k', linestyle= ':')
    #plt.vlines(-0.5, -5, 500, color = 'k', linestyle= ':')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()
    exit()
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