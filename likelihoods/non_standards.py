from tkinter import W
from cobaya.likelihood import Likelihood
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from functools import lru_cache


def cov_log_likelihood(mu_model, mu, cov):
    delta = np.array([mu_model - mu])
    inv_cov = np.linalg.inv(cov)
    deltaT = np.transpose(delta)
    chit2 = np.sum(delta @ inv_cov @ deltaT)
    B = np.sum(delta @ inv_cov)
    C = np.sum(inv_cov)
    chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
    return -0.5*chi2 


@lru_cache(maxsize=4096)
def interp_dl(model, *params):
    z_interp = np.geomspace(0.0001, 2.5, 1000)
    dl_interp = interp1d(z_interp, model(*params), kind="linear")
    return dl_interp


class FLCDMclass(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Load in data
        self.cov_size = int(np.genfromtxt(self.cov_path, comments='#',dtype=None)[0])
        HD = np.genfromtxt(self.HD_path, names=True, comments='#')
        self.cov_arr = np.genfromtxt(self.cov_path, comments='#',dtype=None)[1:]
        self.z_data = HD['zCMB']
        self.mu_data = HD['MU']
        self.mu_error = HD['MUERR']
        cov_reshape = self.cov_arr.reshape(self.cov_size,self.cov_size) 
        mu_diag = np.diag(self.mu_error)**2
        self.cov = mu_diag+cov_reshape
        self.zs = np.geomspace(0.0001, 2.5, 1000) # list of redshifts to interpolate between.


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        dl_interp = interp_dl(self.lum_dist_interp, om)
        dl_data = dl_interp(self.z_data)
        dist_mod = 5 * np.log10(dl_data)
        return cov_log_likelihood(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol):
        Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
        return 1.0 / Hz

    def distmod(self, om):
        ol = 1 - om
        zx = self.z_data
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om):
        ol = 1 - om
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return lum_dist

    def label(self):
        return [r"$\Omega_m$"]

