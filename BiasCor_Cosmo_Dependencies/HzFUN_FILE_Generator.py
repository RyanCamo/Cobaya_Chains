import numpy as np
import sys
from Hzs import *
sys.path.append('/Users/RyanCamo/Desktop/Cobaya/Cobaya_Chains')
from chainconsumer import ChainConsumer
from matplotlib import pyplot as plt
from model_info import *
from chainconsumer import ChainConsumer
from pathlib import Path


# This code takes an input chain and the model used when fitting for that chain.
# extracts the best fit parameters and outputs a 'HzFUN_FILE' which is a sim-input 
# key used to generate mock data.
# The output format is 2 columns: z, H(z) with the first row being z=0.


# This function gets the best fit parameters for the specific chains used.
def get_param(samples, label, weights):
    c = ChainConsumer()
    print('extracting best fit parameters...')
    c.add_chain(samples, parameters=label, linewidth=2.0, weights=weights, name="MCMC", kde=1.5, color="red").configure(summary=True,shade_alpha=0.3,statistics="cumulative")
    #fig = c.plotter.plot(figsize="column")
    #plt.show()
    #exit()
    params = []
    for i, labelx in enumerate(label):
        params.append(c.analysis.get_summary(chains="MCMC")[labelx][1])
    print('the best fit parameters are:')
    print('%s' %params)
    return params

def create_HzFUN(model, chain_path, save_path):

    print('Creating HzFUN_FILE...')

    # Generate a list of redshifts starting with z=0.
    zs = np.geomspace(0.0001, 2.5, 1000)
    z = np.insert(zs,0,0)

    # Get model details
    label, params, legend = get_info(model.__name__)

    # used to extract weights and correct columns
    cols = []
    for i, l in enumerate(label):
        cols.append(i+2)
           
    chain = np.loadtxt(chain_path, usecols=(cols), comments='#')
    weights = np.loadtxt(chain_path, usecols=(0), comments='#')
    params = get_param(chain,label, weights)
    H0 = 70
    Hz = H0*model(z, *params)
    np.savetxt(save_path, np.c_[z, Hz], fmt="%10.4f")
    print('HzFUN_FILE created and saved to: %s' %save_path)
    

if __name__ == "__main__":
    model = FLCDM # model used in the fit
    chain_path = Path('chains/SN_TESTS/FLCDM_31.1.txt')
    save_path = Path('BiasCor_Cosmo_Dependencies/CONV_TEST_HzFUN_FILES/GAL_11.1.txt')

    create_HzFUN(model, chain_path, save_path)

