import numpy as np
from Hzs import *
from model_info import *
from matplotlib import pyplot as plt
from matplotlib import rcParams, rc
from pathlib import Path
from BiasCor_Cosmo_Dependencies.HzFUN_FILE_Generator import get_param
from BiasCor_Cosmo_Dependencies.Models_DL import *


# This file produces a risidual plot in mu. Comparing the two iterations against the True Cosmology.

def get_true_cosmo(z, mock_data):
    if mock_data ==1: 
        cosmo = FLCDM_(z, 0.315)
    elif mock_data ==2:
        cosmo = wCDM_(z, 0.350, 0.600, -1.1)
        #cosmo2 = FLCDM_(z, 0.315)
    elif mock_data ==3:
        cosmo = GChap_(z, 0.65, 0.200, 0.05)
        # A = 0.7, a = 0.1, ok = 0.05
        #cosmo2 = FLCDM_(z, 0.315)
    elif mock_data ==4:
        cosmo = IDEA_(z, 0.25, -1.2, 0.100)
    elif mock_data ==5:
        cosmo = GAL_(z, 0.350, 0.500)
    else:
        print('NOT A VALID MOCK DATASET')
        exit()
    return cosmo#, cosmo2

def get_distmod(z, model, params):
    cosmo = model(z, *params)
    return cosmo

def compare_cosmo(mock_data, model, save_path):
    # File paths for the first and second iteration chains.
    if model.__name__ == 'GChap_':
        first_path = Path('chains/SN_TESTS/GCG_%s1.1.txt' % (mock_data))
        second_path = Path('chains/SN_TESTS/GCG_%s2.1.txt' % (mock_data))
    else:
        first_path = Path('chains/SN_TESTS/%s%s1.1.txt' % (model.__name__, mock_data))
        second_path = Path('chains/SN_TESTS/%s%s2.1.txt' % (model.__name__, mock_data))
        third_path = Path('chains/SN_TESTS/%s%s3.1.txt' % (model.__name__, mock_data))
        fourth_path = Path('chains/SN_TESTS/%s%s4.1.txt' % (model.__name__, mock_data))
    
    # Redshift range to compare
    z = np.geomspace(0.0001, 2.5, 500)

    # The true cosmology
    true_mu = get_true_cosmo(z, mock_data)

    # Fitting model details
    label, params, legend = get_info(model.__name__.strip('_')) 

    # Get the best fit parameters for the first and second iterations

    ### used to extract weights and correct columns of the chain
    cols = []
    for i, l in enumerate(label):
        cols.append(i+2)     
         
    first_chain = np.loadtxt(first_path, usecols=(cols), comments='#')
    first_weights = np.loadtxt(first_path, usecols=(0), comments='#')
    second_chain = np.loadtxt(second_path, usecols=(cols), comments='#')
    second_weights = np.loadtxt(second_path, usecols=(0), comments='#')
    #third_chain = np.loadtxt(third_path, usecols=(cols), comments='#')
    #third_weights = np.loadtxt(third_path, usecols=(0), comments='#')
    #fourth_chain = np.loadtxt(fourth_path, usecols=(cols), comments='#')
    #fourth_weights = np.loadtxt(fourth_path, usecols=(0), comments='#')

    first_params = get_param(first_chain,label, first_weights)
    second_params = get_param(second_chain,label, second_weights)
    #third_params = get_param(third_chain,label, third_weights)
    #fourth_params = get_param(fourth_chain,label, fourth_weights)
    
    # Calculate the distance modulus for the first and second iterations
    first_distmod = get_distmod(z, model, first_params)
    second_distmod = get_distmod(z, model, second_params)
    #third_distmod = get_distmod(z, model, third_params)
    #fourth_distmod = get_distmod(z, model, fourth_params)

    # Create a plot
    plt.rc('font', family='serif')
    rcParams['mathtext.fontset'] = 'dejavuserif'
    fig, ax = plt.subplots(1, 1, sharex=True,figsize=(5,4))
    ax.plot(z, first_distmod-true_mu, label = r'First Iteration', linestyle = ':', color = 'b')
    ax.plot(z, second_distmod-true_mu, label = r'Second Iteration', linestyle = '-.', color = 'y')
    #ax.plot(z, third_distmod-true_mu, label = r'Third Iteration', linestyle = '-.', color = 'c')
    #ax.plot(z, fourth_distmod-true_mu, label = r'Fourth Iteration', linestyle = ':', color = 'y')
    #ax.plot(z, cosmo2-true_mu, label = r'Zero Iteration', linestyle = '-.', color = 'c')
    ax.plot(z, true_mu-true_mu, label = r'True Cosmology', linestyle = '--', color = 'k')
    #ax.text(0.01,-0.07,'%s' % model.__name__.strip('_'), family='serif',color='black',fontsize=12,ha='left') 
    ax.legend()
    ax.set_xlabel('Redshift, z', fontsize=20)
    ax.set_ylabel(r'$\Delta \mu$', fontsize=20)
    plt.tight_layout()
    plt.savefig(save_path)  
    plt.show()

def compare_truths():
    # Redshift range to compare
    z = np.geomspace(0.0001, 2.5, 500)
    # Create a plot
    plt.rc('font', family='serif')
    rcParams['mathtext.fontset'] = 'dejavuserif'
    fig, ax = plt.subplots(1, 1, sharex=True,figsize=(5,4))
    truth = get_true_cosmo(z, 1)
    for i in range(5):
        cosmo = get_true_cosmo(z, i+1)
        plt.plot(z, cosmo-truth, label = 'Mock: %s' % str(i+1))
    #cosmo3 = FLCDM_(z, 0.3275684329632337)
    #ax.plot(z, cosmo3-truth, label='mine')
    ax.set_xlabel('Redshift, z', fontsize=20)
    ax.set_ylabel(r'$\Delta \mu$', fontsize=20)
    ax.legend()
    plt.show()
    #plt.tight_layout()
    #plt.savefig(Path('BiasCor_Cosmo_Dependencies/Comparing_Iterations/Compare_truths.png'))

# Left      - 0.17
# Bottom    - 0.15
# right     - 0.97
# Top       - 0.97

if __name__ == "__main__":
    # What mock data set to compare? (To see parameter values used in these mock sets see get_true_cosmo)
    # FLCDM = 1
    # wCDM = 2
    # GCG = 3
    # IDE1 = 4
    # GAL = 5 

    ##########################
    # Unhash this section to manually compare 2 chains

    # Fitting model details
    #label, params, legend = get_info(FLCDM.__name__) 

    # Get the best fit parameters for the first and second iterations

    ### used to extract weights and correct columns of the chain
    #cols = []
    #for i, l in enumerate(label):
    #    cols.append(i+2)     

    #first_chain = np.loadtxt('chains/2.1.txt', usecols=(cols), comments='#')
    #first_weights = np.loadtxt('chains/2.1.txt', usecols=(0), comments='#')
    #irst_params = get_param(first_chain,label, first_weights)
    #print(first_params)
    #first_chain = np.loadtxt('chains/SN_TESTS/FLCDM_13.1.txt', usecols=(cols), comments='#')
    #first_weights = np.loadtxt('chains/SN_TESTS/FLCDM_13.1.txt', usecols=(0), comments='#')
    #first_params = get_param(first_chain,label, first_weights)
    #print(first_params)
    #exit()

    ##########################

    mock_data = 5
    model = FLCDM_ # which fitting model to compare iterations?
    save_path = Path('BiasCor_Cosmo_Dependencies/Comparing_Iterations/Mock:%s_Fit:%s.png' % (mock_data, model.__name__)) # where to save the figure
    compare_cosmo(mock_data, model, save_path)
    #compare_truths()

