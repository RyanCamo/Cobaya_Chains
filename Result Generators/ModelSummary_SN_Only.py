import numpy as np
import sys
sys.path.append('Cobaya_Chains')
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt
from model_info import *
import scipy.special as sc

### WHEN LOOKING FOR SN2 CONSTRAINTS MAKE SURE TO CHANGE THE MODEL TAG IN likelihoods_SN_bias.py - LINE 19
### EACH MODEL DONE 1 AT A TIME.

# This prints of the details for a particular model that is used in the summary tables.
# This code by default is used for my specific output chains file paths and naming conventions


# This function gets the best fit parameters for the specific chains used.
def get_param(samples, label, weights):
    c = ChainConsumer()
    c.add_chain(samples[burnin:], parameters=label, linewidth=2.0, weights=weights[burnin:], name="MCMC", kde=1.5, color="red").configure(summary=True,shade_alpha=0.3,statistics="cumulative")
    params = []
    params_upper = []
    params_lower = []
    for i, labelx in enumerate(label):
        print(c.analysis.get_summary(chains="MCMC")[labelx])
        params_upper.append(c.analysis.get_summary(chains="MCMC")[labelx][2] - c.analysis.get_summary(chains="MCMC")[labelx][1])
        params_lower.append(c.analysis.get_summary(chains="MCMC")[labelx][1] - c.analysis.get_summary(chains="MCMC")[labelx][0])
        params.append(c.analysis.get_summary(chains="MCMC")[labelx][1])
    return params, params_upper, params_lower

# This function loads in the chain and then gets the best fit params from get_params.
def get_bestparams(model,label):
    # Makes a note of which columns to use.
    cols = []
    for i, l in enumerate(label):
        cols.append(i+2)

    if SN1 == 1 and SN2 == 0:
        main_chain = np.loadtxt('Cobaya_Chains/chains/SN/%s_SN.1.txt' %(model), usecols=(cols), comments='#')
        main_chain_weights = np.loadtxt('Cobaya_Chains/chains/SN/%s_SN.1.txt' %(model), usecols=(0), comments='#')
    elif SN1 == 0 and SN2 == 1:
        main_chain =  np.loadtxt('Cobaya_Chains/chains/SN_BiasCor/%s_SN_2.1.txt' %(model), usecols=(cols), comments='#') 
        main_chain_weights =  np.loadtxt('Cobaya_Chains/chains/SN_BiasCor/%s_SN_2.1.txt' %(model), usecols=(0), comments='#') 
    else:
        print('Not valid table options..')
        exit()

    params_main, params_main_upper, params_main_lower = get_param(main_chain,label, main_chain_weights)
    return params_main, params_main_upper, params_main_lower

def get_summary(model, dataL):
    ## Prints a list of relevant data for Summary Table.
    # chi^2/dof
    # GoF (%)
    # Parameter constraints for SN-1 or SN-2
    label, begin, legend = get_info(model.__name__)
    if SN2 == 1: 
        dataL = int(np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains/BiasCor_Cosmo_Dependencies/PIPPIN_OUTPUTS/%s/MOD_%s_cov.txt" % (model.__name__, model.__name__), comments='#',dtype=None)[0])
    else:
        dataL=dataL
    params_main, params_main_upper, params_main_lower = get_bestparams(model.__name__,label)
    params_main = np.array(params_main)
    p_main = model(*params_main)
    chi2_main = -p_main/0.5
    free_params = len(label)
    dof = dataL - free_params
    #Calculating GoF (%)
    GoF = 100*sc.gammaincc(0.5*dof, 0.5*chi2_main)

    # Calculating AIC/BIC
    AIC = -2*p_main + 2*free_params - refAIC
    BIC = -2*p_main + free_params*np.log(dataL) - refBIC

    print(r'%s  & $%s/%s$ & $%s$ & %s & %s    \\' % (model.__name__,round(chi2_main,1) , dof, round(GoF,1), round(AIC,1), round(BIC,1)))  
    for i, param in enumerate(params_main):
        print(r'%s : %s^{+%s}_{-%s}' %(label[i], np.round(param,3), np.round(params_main_upper[i],3), np.round(params_main_lower[i],3)))



if __name__ == "__main__":

    # First we need to indicate what data are after.

    ## TABLE OPTIONS: 1 = True, 0 = False - ONLY SELECT 1 
    SN1 = 0 # Get SN-1 Only Constraints - Before BiasCor Sim changes
    SN2 = 1 # Get SN-2 Only Constraints - After BiasCor Sim changes

    # amount of data points per dataset
    SN1_data = 1891

    if SN1 == 1 and SN2 == 0:
        from likelihoods_SN import *
        dataL = SN1_data
    elif SN1 == 1 and SN2 == 1:
        print('ERROR: ONLY 1 SET OF CONSTRAINTS AT A TIME')
        exit()
    elif SN1 == 0 and SN2 == 1:
        from likelihoods_SN_bias import *
        dataL = 0
        print('Make Sure line 19 in likelihoods_SN_bias.py matches the model you want constraints for!!')
    else:
        print('Not valid table options.')
        exit()

    if SN2 ==1:
        refAIC = 1885.4922812247555 # Manually calculated for Mock dataset
        refBIC = 1891.0387274985014 # Manually calculated for Mock dataset

    if SN1 ==1:
        refAIC = 1880.569167491951 # Manually calculated for Mock dataset
        refBIC = 1886.1140285606095 # Manually calculated for Mock dataset


    # Get 1 Model at a time: FOR SN2 constraints each time the likelihoods_SN_bias.py file needs the model changing!!
    #burnin
    burnin = 0
    get_summary(IDEB, dataL)