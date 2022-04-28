import numpy as np
import sys
sys.path.append('Cobaya_Chains')
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt
from likelihoods_BAO_CMB_SN import *
from model_info import *
import scipy.special as sc

### WHEN LOOKING FOR CONSTRAINTS MAKE SURE THE MODEL IN LINE 17 OF likelihoods_BAO_CMB_SN.py 
# MATCHES THE MODEL YOU WANT CONSTRAINTS FOR 

# This prints of the details for a particular model that is used in the summary tables.
# This code by default is used for my specific output chains file paths and naming conventions


# This function gets the best fit parameters for the specific chains used.
def get_param(samples, label, weights):
    c = ChainConsumer()
    c.add_chain(samples[burnin:], parameters=label, linewidth=2.0, name="MCMC", weights=weights[burnin:], kde=1.5, color="red").configure(summary=True,shade_alpha=0.3,statistics="cumulative")
    params = []
    params_upper = []
    params_lower = []
    for i, labelx in enumerate(label):
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
    main_chain = np.loadtxt('Cobaya_Chains/chains/CMB+BAO+SN/%s_CMB_BAO_SN.1.txt' %(model), usecols=(cols), comments='#')
    weights = np.loadtxt('Cobaya_Chains/chains/CMB+BAO+SN/%s_CMB_BAO_SN.1.txt' %(model), usecols=(0), comments='#')
    params_main, params_main_upper, params_main_lower = get_param(main_chain,label, weights)
    return params_main, params_main_upper, params_main_lower

def get_summary(model, CMB_BAO_dataL):
    ## Prints a list of relevant data for Summary Table.
    # chi^2/dof
    # GoF (%)
    # Parameter constraints for SN-1 or SN-2
    label, begin, legend = get_info(model.__name__)
    SN_dataL = int(np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains/BiasCor_Cosmo_Dependencies/PIPPIN_OUTPUTS/%s/MOD_%s_cov.txt" % (model.__name__, model.__name__), comments='#',dtype=None)[0])
    dataL = SN_dataL + CMB_BAO_dataL
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
    print(AIC)
    print(BIC)

    print(r'%s  & $%s/%s$ & $%s$ & %s & %s    \\' % (model.__name__,round(chi2_main,1) , dof, round(GoF,1), round(AIC,1), round(BIC,1))) 
    for i, param in enumerate(params_main):
        print(r'%s : %s^{+%s}_{-%s}' %(label[i], np.round(param,3), np.round(params_main_upper[i],3), np.round(params_main_lower[i],3)))



if __name__ == "__main__":

    # amount of data points per dataset
    CMB_BAO_dataL = 14

    # The AIC/BIC valyes for FLCDM here as reference values
    refAIC = 1898.1576586232297
    refBIC = 1903.711469475238
    # Get 1 Model at a time: FOR constraints each time the likelihoods_SN_CMB_BAO.py file needs the model changing!!
    burnin=1000
    get_summary(GChap, CMB_BAO_dataL)