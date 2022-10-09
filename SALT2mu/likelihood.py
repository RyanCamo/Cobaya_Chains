from re import S
from cobaya.likelihood import Likelihood
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from functools import lru_cache
from pathlib import Path
import pandas as pd

def curv(ok, x):
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
        return D
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
        return D
    else:
        D = x
        return D

class SALT2mu(Likelihood):

    def initialize(self):

        # NOTE: These files should be the input into BBC not the output:
        
        # Load in the data
        self.dfFITRES = pd.read_csv(self.FITRES_path, delim_whitespace=True, comment="#")

        # Load in Fiducial parameters
        self.fid_om = self.p1
        self.fid_ol = self.p2
        self.fid_w = self.p3
        self.fid_h0 = self.p4


    #TODO: not calculating any bias corrections and where is the alpha/beta grid?? - Just grabbing from the output file..
    def logp(self, **params_values):
        alpha = params_values['alpha']
        beta = params_values['beta']
        sig_int = params_values['sig_int'] # NOTE: I could put my intrinsic scatter model here.

        # This is currently set up using 2 bins. 0 - 0.1 and 0.1 - 1.5.
        # For 20 bins: [0.010, 0.031, 0.052, 0.076, 0.101, 0.127, 0.156, 0.187, 0.221, 0.258, 0.298, 0.343, 0.392, 0.447, 0.510, 0.581, 0.664, 0.760, 0.877, 1.019, 1.200] 
        bins = [0.010, 0.031, 0.052, 0.076, 0.101, 0.127, 0.156, 0.187, 0.221, 0.258, 0.298, 0.343, 0.392, 0.447, 0.510, 0.581, 0.664, 0.760, 0.877, 1.019, 1.200] 
        #bins = #[0, 1.5]#[0, 0.1, 1.5] # Define bins, could be done better...
        self.dfFITRES['bins'] = np.digitize(self.dfFITRES['zHD'].values, bins) # adds an index to the data defining what bin 
        num_bins = len(bins) - 1 
        chi2_tmp = [] # temp stores the chi2 values for each bin

        # Loops through bins and calculates chi2 then adds them together later.
        for b in range(num_bins):
            b = b + 1
            # Calculate Fiducial cosmology over the bin range.
            df = self.dfFITRES.loc[self.dfFITRES['bins'] == b]
            fid_cosmo = self.distmod(df['zHD'].values, self.fid_om, self.fid_ol, self.fid_w) + 10
            M0DIF = params_values['M0DIF%s' % b]
            mu_data = -2.5*np.log10(df['x0'].values) + alpha * (df['x1'].values) - beta * (df['c'].values) - df['biasCor_mu'] - M0DIF

            # calculating the term in the denominator from Kessler 2017 Eq. 3

            # NOTE: No intrinsic scatter model. Using the already calculated error on zHD instead of calculating myself (hashed out)
            sigma_lens2 = (0.055*df['zHD'].values)**2
            sigma_z2 = df['zHDERR'].values**2 #(5/(np.log(10))) * ( (1+self.dfFITRES['zHD'].values) / (self.dfFITRES['zHD'].values*(1+(self.dfFITRES['zHD'].values/2))) ) * np.sqrt(self.dfFITRES['zHDERR'].values**2 + (self.dfFITRES['VPECERR'].values/self.c)**2)
            sigma_mu2 = sig_int**2  + sigma_lens2 + sigma_z2 + df['x0ERR'].values**2 + (alpha**2)*df['x1ERR'].values**2 + (beta**2)*df['cERR'].values**2 + 2*alpha*df['COV_x1_x0'].values - 2*beta*df['COV_c_x0'].values - 2*alpha*beta*df['COV_x1_c'].values
            chi2_tmp.append(np.sum(((mu_data - fid_cosmo)**2)/sigma_mu2 + 2*np.log(np.sqrt(sigma_mu2)))) #- guassian normalization term required if using bias corrections

            #G10 = (0.7*sig_int**2 + (beta**2)*0.3*sig_int**2 - 0.21*beta*sig_int) # my attempt at G10 scatter model.. didnt work.

        chi2 = np.sum(chi2_tmp) 
        return -0.5*chi2 

    def Hz_inverse(self, z, om, ol, w):
        self.c = 299792458
        ok = 1.0 - om - ol
        Hz = (self.fid_h0/self.c)*np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
        return 1.0 / Hz

    def distmod(self, zx, om, ol, w):
        ok = 1.0 - om - ol
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx])
        D = curv(ok, x)
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 


if __name__ == "__main__":
    FITRES_path = '/Users/uqrcamil/Documents/RyanCamo/PhD/project.non-standard.models/PIPPIN/PIPPIN_OUTPUTS/BBC_outputs/nonstandard.FITRES'
    dfFITRES = pd.read_csv(FITRES_path, delim_whitespace=True, comment="#")
    df = dfFITRES[["zHEL", "zHELERR", "zCMB", "zCMBERR", "zHD", "zHDERR", "VPEC", "VPECERR"]]
    print(df.loc[0,:])
    print((1+df.loc[0,'zCMB'])*(1+df.loc[0, 'VPEC'])-1)
    c = 299792458
    #x = (5/(np.log(10))) * ( (1+dfFITRES.loc[0,'zCMB']) / (dfFITRES.loc[0,'zCMB']*(1+(dfFITRES.loc[0,'zCMB']/2))) ) * np.sqrt(dfFITRES.loc[0,'zCMBERR']**2 + (dfFITRES.loc[0,'VPECERR']*1e3/c)**2)
    #x = (5/(np.log(10))) * ( (1+dfFITRES.loc[0,'zCMB']) / (dfFITRES.loc[0,'zCMB']*(1+(dfFITRES.loc[0,'zCMB']/2))) ) * np.sqrt(dfFITRES.loc[0,'zCMBERR']**2) +  (0.055*dfFITRES.loc[0,'zCMBERR'])**2
    x = np.sqrt(dfFITRES.loc[0,'zCMBERR']**2 + ((1+dfFITRES.loc[0,'zCMB'])*(dfFITRES.loc[0,'VPECERR']*1e3/c))**2)
    print(x)