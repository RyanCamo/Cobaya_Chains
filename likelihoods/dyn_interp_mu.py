from cProfile import label
import numpy as np
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import quad
from matplotlib.ticker import ScalarFormatter


# This file contains the function that dynamically interpolates mu

#NOTE: currently works but is doing this unbinned. Is this acceptable?
#TODO: better way to make sure data sets are the same


c = 299792458

def Hz_inverse(z, om, ox, w):
    """ Calculate 1/H(z). Will integrate this function. """
    ok = 1.0 - om - ox
    Hz = np.sqrt(om*(1+z)**3 + ox*(1+z)**(3*(1+w)) + ok*(1+z)**2)
    return 1.0 / Hz

def dist_mod(zs, om, ox, w):
    """ Calculate the distance modulus, correcting for curvature"""
    ok = 1.0 - om - ox
    x = np.array([quad(Hz_inverse, 0, z, args=(om, ox, w))[0] for z in zs])
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
    else:
        D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    return dist_mod

def dist_milne(zs, om, ox, w):
    """ Calculate the distance modulus, correcting for curvature"""
    ok = 0.0
    x = np.array([quad(Hz_inverse, 0, z, args=(om, ox, w))[0] for z in zs])
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
    else:
        D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    return dist_mod

def dyn_mu(mu_model, df0, df1, df2):
    # 0 = NOM
    # 1 = HIGH
    # 2 = LOW
    
    # Calculates the exact cosmologies used for the 3x BiasCor sims
    df0['muSIM0'] = dist_mod(df0['zCMB'], 0.3, 0.7, -1)
    df0['muSIM1'] = df0['muSIM0'] + 0.05
    df0['muSIM2'] = df0['muSIM0'] - 0.05 

    df0['muBBC'] = np.zeros(len(df0['zCMB']))

    #implementing interpolation, however it will not explore outside the range of the 3x BiasCor sims above.
    for i, mu in enumerate(mu_model): 
        if (mu_model[i] >= df0.loc[i, 'muSIM0']) & (mu_model[i] <= df0.loc[i, 'muSIM1']):
            f_int = ( (mu - df0.loc[i, 'muSIM0']) / (df0.loc[i, 'muSIM1'] - df0.loc[i, 'muSIM0']) ) 
            df0.loc[i,'muBBC'] = df0.loc[i,'MU'] + f_int*( df1.loc[i,'MU'] - df0.loc[i,'MU'] )
        if mu_model[i] > df0.loc[i, 'muSIM1']: 
            df0.loc[i,'muBBC'] = df1.loc[i,'MU']
        if (mu_model[i] <= df0.loc[i, 'muSIM0']) &  (mu_model[i] >= df0.loc[i, 'muSIM2']):
            f_int = ( (mu - df0.loc[i, 'muSIM0']) / (df0.loc[i, 'muSIM2'] - df0.loc[i, 'muSIM0']) )
            df0.loc[i,'muBBC'] = df0.loc[i,'MU'] + f_int*( df2.loc[i,'MU'] - df0.loc[i,'MU'] )
        if mu_model[i] < df0.loc[i, 'muSIM2']:
            df0.loc[i,'muBBC'] = df2.loc[i,'MU']
    return df0['muBBC']


if __name__ == "__main__":
    # Load in the 3x Hubble diagrams
    HIGH_path = '/Users/uqrcamil/Documents/GitHub/RyanCamosRepo/midway/outputs/RC_3HD/WIDE/HIGH/hubble_diagram.txt'
    df1= pd.read_csv(HIGH_path, delim_whitespace=True, comment="#")
    NOM_path = '/Users/uqrcamil/Documents/GitHub/RyanCamosRepo/midway/outputs/RC_3HD/7_CREATE_COV/CCNOMNS_BBCNOMNS/output/hubble_diagram.txt'
    df0 = pd.read_csv(NOM_path, delim_whitespace=True, comment="#")
    LOW_path = '/Users/uqrcamil/Documents/GitHub/RyanCamosRepo/midway/outputs/RC_3HD/WIDE/LOW/hubble_diagram.txt'
    df2 = pd.read_csv(LOW_path, delim_whitespace=True, comment="#")

    # Removes all odd SN. We loose 14 SN from the nominal analysis
    df1 = df1[df1['CID'].isin(df0['CID'])]
    df1 = df1[df1['CID'].isin(df2['CID'])]
    df2 = df2[df2['CID'].isin(df1['CID'])]
    df2 = df2[df2['CID'].isin(df0['CID'])]
    df0 = df0[df0['CID'].isin(df2['CID'])]
    df0 = df0[df0['CID'].isin(df1['CID'])]
    df0.reset_index(drop=True, inplace=True)
    df1.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    mu_model = dist_mod(df0['zCMB'], 0.5, 0.5, -1) + 10 ## Representing representing 1 step in the fitting procedure

    mu_data = dyn_mu(mu_model, df0, df1, df2) # Representing the corrected distance modulus after interpolating
    
    plt.errorbar(df0['zCMB'].values, mu_data - df0['MU'], fmt='.', color='b', capsize=1, elinewidth=1, alpha=0.2) # Difference from the nominal
    plt.xlabel('Redshift, $z$', fontsize =16)
    plt.ylabel('$\Delta \mu$ to nominal Hubble diagram', fontsize =16)
    plt.show()


    ## PLOT OF mu_residual - saved in dyn_interp/, just looking at the difference in BiasCor 
    plt.errorbar(df1['zCMB'].values, df1['MU'].values - df0['MU'].values, fmt='.', color='r', capsize=1, elinewidth=1, alpha=0.2, label=r'$\Delta_{ \mu } = 0.5$')
    plt.errorbar(df2['zCMB'].values, df2['MU'].values - df0['MU'].values, fmt='.', color='b', capsize=1, elinewidth=1, alpha=0.2, label=r'$\Delta_{ \mu } = -0.5$')
    plt.ticklabel_format(style='plain')
    plt.xlabel('Redshift, $z$', fontsize =16)
    plt.ylabel('$\mu$ residual to nominal Hubble diagram', fontsize =16)
    plt.legend(loc='lower left', fontsize =14)
    plt.ylim(-.1, .1)
    plt.show()

    df0['muSIM0'] = dist_mod(df0['zCMB'], 0.3, 0.7, -1) + 10
    df0['muSIM1'] = df0['muSIM0'] + 0.5
    df0['muSIM2'] = df0['muSIM0'] - 0.5 
    milne = dist_milne(df0['zCMB'], 0, 0 , -1) + 10
    #PLOT of different models
    plt.plot(df0['zCMB'], mu_model-milne, 'r--')
    #plt.plot(df0['zCMB'], mu_data, 'b')
    plt.plot(df0['zCMB'], df0['muSIM0']-milne, 'k')
    plt.plot(df0['zCMB'], df0['muSIM1']-milne, 'k')
    plt.plot(df0['zCMB'], df0['muSIM2']-milne, 'k')
    plt.xlabel('Redshift, $z$', fontsize =16)
    plt.ylabel('$\mu$ residual to an empty universe', fontsize =16)
    plt.show()
