import numpy as np
import sys
sys.path.append('Cobaya_Chains')
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt
from model_info import *
import scipy.special as sc

### THIS CAN ONLY BE USED ON MODELS THAT ALREADY HAVE CONTOURS MADE. THIS ENSURES THE SAME BURNIN IS USED
### FOR THE BEST FITS.

# Table generator to get the model summary
# This table generator by default is used for my specific output chains file paths and naming conventions
# and defaults to DES5YR_UNBIN for SN data. If chains are to be loaded in from somewhere else, be sure to 
# change the file paths in lines 45, 47, 49 and the file name is required.

# First we need to indicate what data we are using to constrain our models. This will change the table.

## TABLE OPTIONS: 1 = True, 0 = False
SN = 1 # Use SN data in the model comparison table
CMB_BAO = 1 # Use CMB_BAO data in the model comparison table


# amount of data points per dataset
SN_data = 1891
CMB_BAO_data = 14

if SN == 1 and CMB_BAO == 1:
    from likelihoods_BAO_CMB_SN import *
    dataL = SN_data + CMB_BAO_data
elif SN == 1 and CMB_BAO == 0:
    from likelihoods_SN import *
    dataL = SN_data
elif SN == 0 and CMB_BAO == 1:
    from likelihoods_BAO_CMB import *
    dataL = CMB_BAO_data
else:
    print('Not valid table options.')
    exit()


# This function gets the best fit parameters for the specific chains used.
def get_param(samples, label, model, burn):
    c = ChainConsumer()
    c.add_chain(samples[burn:], parameters=label, linewidth=2.0, name="MCMC", kde=1.5, color="red").configure(summary=True,shade_alpha=0.3,statistics="cumulative")
    params = []
    for i, labelx in enumerate(label):
        params.append(c.analysis.get_summary(chains="MCMC")[labelx][1])
    return params

# This function loads in the chain and then gets the best fit params from get_params.
def get_bestparams(model,label):
    # Makes a note of which columns to use.
    cols = []
    for i, l in enumerate(label):
        cols.append(i+2)

    ## TAKING INTO CONSIDERATION THE CORRECT BURN IN. TO OVERWRITE UNCOMMENT #burn = 0.
    burnSN, burnBAO_CMB, burnBAO_CMB_SN  = [0,0,0] #np.loadtxt('Cobaya_Chains/Contours/OUTPUT/BURNIN/%s_Burnin.txt' % (model))
    #Cobaya_Chains/chains/CMB+BAO
    if SN == 1 and CMB_BAO == 1:
        samples_tot = np.loadtxt('Cobaya_Chains/chains/CMB+BAO+SN/%s_CMB_BAO_SN.1.txt' %(model), usecols=(cols), comments='#')
        samples_SN = np.loadtxt('Cobaya_Chains/chains/SN/%s_SN.1.txt' %(model), usecols=(cols), comments='#')
        burn = burnBAO_CMB_SN
    elif SN == 1 and CMB_BAO == 0:
        samples = np.loadtxt('Cobaya_Chains/chains/SN/DES5YR/UNBIN/%s_DES5YR_UNBIN.1.txt' %(model), usecols=(cols), comments='#')
        burn = burnSN
    elif SN == 0 and CMB_BAO == 1:
        samples = np.loadtxt('Cobaya_Chains/chains/CMB+BAO/%s_CMB_BAO.1.txt' %(model), usecols=(cols), comments='#')
        burn = burnBAO_CMB
    else:
        print('Not valid table options..')
        exit()

    #burn = 0 # Uncomment to overwrite
    params = get_param(samples,label,model, burn)
    print(params)
    return params 

def get_table(model):
    ## Printing the Table for each model
    print(r'\begin{table}')
    print(r' \centering')
    print(r'  \begin{threeparttable}')      
    print(r' \caption{\textsc{Summary of the results for %s }}' % (model.__name__))        
    print(r' \begin{tabular}{c l c c c c c c}')    
    print(r'     \hline \hline')
    # need to loop through params and print header 
    print(r'   & Data Sets & $\Omega_m$ & $\Omega_{\Lambda}$ & $\chi^2/\text{dof}$ & GoF (\%) & $\Delta \Omega_m$ & $\Delta \Omega_{\Lambda}$  \\')       
    print(r'     Model & $\chi^2$/dof & GoF (\%) & $\Delta$AIC & $\Delta$BIC \\')         
    print(r'     \hline')    
    for i, model in enumerate(models):
        # Calculating # AIC & BIC
        label, begin, legend = get_info(model.__name__)
        params = get_bestparams(model.__name__,label)
        params = np.array(params)
        #if model.__name__ == 'FCa':
        #    params = [0.3, 0.7, 0]
        p1 = model(*params)
        chi2 = -p1/0.5
        free_params = len(label)
        dof = dataL - free_params
        #Calculating GoF (%)
        GoF = 100*sc.gammaincc(0.5*dof, 0.5*chi2)

        #print(chi2/dof)
        #refAIC = -2*np.max(p2) + 2#*free_params # may need to set LCDM as reference but may be able to define one based on the max/min in a list
        #refBIC = -2*np.max(p2) + np.log(n)#free_params*np.log(n)
        if model.__name__ == 'FLCDM':
            refAIC = -2*np.max(p1) + 2*free_params # may need to set LCDM as reference but may be able to define one based on the max/min in a list
            refBIC = -2*np.max(p1) + free_params*np.log(dataL)
        AIC = -2*p1 + 2*free_params - refAIC
        BIC = -2*p1 + free_params*np.log(dataL) - refBIC
        print(r'        %s  & $%s/%s$ & $%s$ &  %s    &   %s    \\' % (model.__name__,round(chi2,1) , dof, round(GoF,1), round(AIC,1), round(BIC,1)))             
    print(r'     \hline')        
    print(r' \end{tabular}')    
    print(r'\end{table}')

if __name__ == "__main__":
    # Models to Create Data from
    models = [FLCDM, FCa]
    #[FLCDM, LCDM, FwCDM, wCDM, IDE1, IDE2, IDE4, FGChap, GChap, FCa, Fwa, Fwz, DGP, GAL, NGCG]
    get_table(models)