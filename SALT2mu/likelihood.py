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


    # NOTE: I am not calculating any bias corrections, just grabbing the correct ones from the output file..
    #       Therefore they are the correct bias corrections for {z, x0, x1, c} but im applying the wrong 
    #       bias corrections as I sample alpha/beta....  Thought this would bias me to the correct alpha/beta though..?

    def logp(self, **params_values):
        # Sampled parameters from SALT2mu.yaml (the binned M0DIFS are called in the bin loop.
        alpha = params_values['alpha']
        beta = params_values['beta']
        sig_int = params_values['sig_int'] 

        # This is currently set up using 20 bins. Same as BBC output. (To run a quick MCMC, reduce the bin size...)
        # For 20 bins: [0.010, 0.031, 0.052, 0.076, 0.101, 0.127, 0.156, 0.187, 0.221, 0.258, 0.298, 0.343, 0.392, 0.447, 0.510, 0.581, 0.664, 0.760, 0.877, 1.019, 1.200] 
        # For 1 bin: [0, 1.5] & For 2 bins: [0, 0.1, 1.5]
        bins = [0, 0.1, 1.5]
        #bins = [0.010, 0.031, 0.052, 0.076, 0.101, 0.127, 0.156, 0.187, 0.221, 0.258, 0.298, 0.343, 0.392, 0.447, 0.510, 0.581, 0.664, 0.760, 0.877, 1.019, 1.200] 
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

            # calculating sigma_mu^2, the term in the denominator from Kessler 2017 Eq. 3, this varies depending on the paper and intrinsic scatter model.

            sigma_lens2 = (0.055*df['zHD'].values)**2 # I included the lensing term here but wasnt included in Kessler 2017 Eq. 3. Check Pantheon+ paper

            # NOTE: No intrinsic scatter model and im using the already calculated error on zHD instead of calculating myself (hashed out)
            #       This should be ok as the uncertainty on the redshift shouldnt change as I sample alpha/beta/M0DIF

            sigma_z2 = df['zHDERR'].values**2 #(5/(np.log(10))) * ( (1+self.dfFITRES['zHD'].values) / (self.dfFITRES['zHD'].values*(1+(self.dfFITRES['zHD'].values/2))) ) * np.sqrt(self.dfFITRES['zHDERR'].values**2 + (self.dfFITRES['VPECERR'].values/self.c)**2)
            sigma_mu2 = sig_int**2 - (sig_int*beta*df['zHDERR'].values)**2  + sigma_lens2 + sigma_z2 + df['x0ERR'].values**2 + (alpha**2)*df['x1ERR'].values**2 + (beta**2)*df['cERR'].values**2 + 2*alpha*df['COV_x1_x0'].values - 2*beta*df['COV_c_x0'].values - 2*alpha*beta*df['COV_x1_c'].values
            chi2_tmp.append(np.sum(((mu_data - fid_cosmo)**2)/sigma_mu2 + 2*np.log(np.sqrt(sigma_mu2)) - 5)) #- guassian normalization term required if using bias corrections

            #G10 = sig_int**2 - (sig_int*beta*df['zHDERR'].values)**2 my attempt at simplified G10 scatter model, recoved sig_int quite well

        # chi2 first term should = 1*N_data = 1796. Penality term 2*log(typical uncertainty)*N = 2*log(0.1)*1796 = -8270. TOT = -6470
        loglike = -0.5*np.sum(chi2_tmp) 
        return loglike

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
    # Testing things...
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