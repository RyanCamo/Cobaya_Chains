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

def log_likelihood(mu_model, mu, cov): 
    delta = mu_model - mu
    chit2 = np.sum(delta**2 / cov**2)
    B = np.sum(delta/cov**2)
    C = np.sum(1/cov**2)
    chi2 = chit2 - (B**2 / C) + np.log(C/(2* np.pi))
    return -0.5*chi2

@lru_cache(maxsize=4096)
def interp_dl(model, *params):
    z_interp = np.geomspace(0.0001, 2.5, 1000)
    dl_interp = interp1d(z_interp, model(*params), kind="linear")
    return dl_interp


class FLCDM(Likelihood):

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om)      

        return self.like_func(dist_mod, self.mu_data, self.cov)

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
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$"]


class LCDM(Likelihood):

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        ol = params_values['ol']
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol)      

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol):
        Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
        return 1.0 / Hz

    def distmod(self, om, ol):
        zx = self.z_data
        ok = 1.0 - om - ol
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol):
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$"]


class FwCDM(Likelihood):

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        w = params_values['w']
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, w)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, w)      

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w):
        Hz = np.sqrt((om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
        return 1.0 / Hz

    def distmod(self, om, w):
        ol = 1 - om
        zx = self.z_data
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, w):
        ol = 1 - om
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_m$",r"$w$"]

class wCDM(Likelihood):

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        ol = params_values['ol']
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol, w)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol, w)      

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w):
        ok = 1.0 - om - ol
        Hz = np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
        return 1.0 / Hz

    def distmod(self, om, ol, w):
        zx = self.z_data
        ok = 1.0 - om - ol
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol, w):
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$", r"$w$"]

class Fwa(Likelihood):

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        w0 = params_values['w0']
        wa = params_values['wa'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, w0, wa)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, w0, wa)      

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, w0, wa):
        ol = 1.0 - om
        Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0+wa))) * (np.exp(-3*wa*(1-((1+z)**(-1))))) ) )
        return 1.0 / Hz

    def distmod(self, om, w0, wa):
        zx = self.z_data
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, w0, wa))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, w0, wa):
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, w0, wa))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$w_0$", r"$w_a$"]

class Fwz(Likelihood): 

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        w0 = params_values['w0']
        wz = params_values['wz'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, w0, wz) 
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, w0, wz) 

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, w0, wz): 
        ol = 1.0 - om 
        Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0-wz))) * (np.exp(3*wz*z)) ) )
        return 1.0 / Hz

    def distmod(self, om, w0, wz): 
        zx = self.z_data
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, w0, wz))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, w0, wz): 
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, w0, wz))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$w_0$", r"$w_z$"]

class IDEA(Likelihood): 

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        cdm = params_values['cdm']  
        w = params_values['w']
        e = params_values['e'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, cdm, w, e) 
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(cdm, w, e) 

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, cdm, ol, w, e): 
        Hz = np.sqrt(cdm*(1+z)**3 + ol*( ((e)/(w+e))*(1+z)**3 + ((w)/(w+e))*(1+z)**(3*(1+w+e))  ))
        return 1.0 / Hz

    def distmod(self, cdm, w, e):
        zx = self.z_data
        ol = 1.0 - cdm 
        x = np.array([quad(self.Hz_inverse, 0, z, args=(cdm, ol, w, e))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, cdm, w, e): 
        ol = 1.0 - cdm
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(cdm, ol, w, e))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{CDM}$", r"$\omega$", r"$\varepsilon$"]

class IDEB(Likelihood):

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        cdm = params_values['cdm']
        ob = params_values['ob']
        w = params_values['w']
        e = params_values['e'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, cdm, ob, w, e)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(cdm, ob, w, e)

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, cdm, ol, w, e, ob): 
        Hz = np.sqrt(ob*(1+z)**(3)+ ol*(1+z)**(3*(1+w)) + cdm*(((e)/(w+e))*(1+z)**(3*(1+w))  + ((w)/(w+e))*(1+z)**(3*(1-e))))
        return 1.0 / Hz

    def distmod(self, cdm, ob, w, e): 
        zx = self.z_data
        ol = 1 - ob - cdm
        x = np.array([quad(self.Hz_inverse, 0, z, args=(cdm, ol, w, e, ob))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, cdm, ob, w, e): 
        ol = 1 - ob - cdm
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(cdm, ol, w, e, ob))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{CDM}$", r"$\Omega_{b}$", r"$\omega$", r"$\varepsilon$"]

class IDEC(Likelihood): 

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        cdm = params_values['cdm']
        ob = params_values['ob']
        w = params_values['w'] 
        e = params_values['e'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, cdm, ob, w, e) 
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(cdm, ob, w, e) 

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, cdm, ol, w, e, ob): 
        constC = ((cdm)/(ol+cdm) + ((ol)/(ol+cdm))*(1+z)**(3*(w+e)))**(-(e)/(w+e))
        Hz = np.sqrt( ob*(1+z)**3 + cdm*constC*(1+z)**3 +  ol*constC*(1+z)**(3*(1+w+e))) 
        return 1.0 / Hz

    def distmod(self, cdm, ob, w, e): 
        zx = self.z_data
        ol = 1.0 - cdm - ob 
        x = np.array([quad(self.Hz_inverse, 0, z, args=(cdm, ol, w, e, ob))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, cdm, ob, w, e): 
        ol = 1.0 - cdm - ob
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(cdm, ol, w, e, ob))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{CDM}$", r"$\Omega_{b}$", r"$\omega$", r"$\varepsilon$"]

class MPC(Likelihood):

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        q = params_values['q']
        n = params_values['n'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, q, n)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, q, n)

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, q, n):
        Hz = np.sqrt( (om*((z+1)**3))*(1+(((om**(-q))-1)*((z+1)**(3*q*(n-1) ))))**(1/q)  )
        return 1.0 / Hz

    def distmod(self, om, q, n):
        zx = self.z_data
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, q, n))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, q, n): 
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, q, n))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_m$",r"$q$",r"$n$"]

class SCG(Likelihood):

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        A = params_values['A'] 
        ok = params_values['ok']
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, A, ok)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(A, ok)

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, A, ok):
        Hz = np.sqrt(ok*((1+z)**2)+(1-ok)*np.sqrt(A + (1-A)*((1+z)**6)))
        return 1.0 / Hz

    def distmod(self, A, ok):
        zx = self.z_data
        x = np.array([quad(self.Hz_inverse, 0, z, args=(A, ok))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, A, ok):
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(A, ok))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$A$", r"$\Omega_{\text{k}}$"]

class FGCG(Likelihood):

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        A = params_values['A']
        a = params_values['a']
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, A, a)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(A, a)

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, A, a): 
        Hz = np.sqrt((A + (1-A)*((1+z)**(3*(1+a))))**(1.0/(1+a)))
        return 1.0 / Hz

    def distmod(self, A, a):
        zx = self.z_data
        x = np.array([quad(self.Hz_inverse, 0, z, args=(A, a))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, A, a): 
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(A, a))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$A$", r"$\alpha$"]

class GCG(Likelihood): 

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        A = params_values['A'] 
        a = params_values['a']
        ok = params_values['ok'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, A, a, ok) 
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(A, a, ok) 

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, A, a, ok): 
        Hz = np.sqrt((ok*((1+z)**2)) + (1-ok)*(A + (1-A)*((1+z)**(3*(1+a))))**(1/(1+a)))
        return 1.0 / Hz

    def distmod(self, A, a, ok): 
        zx = self.z_data
        x = np.array([quad(self.Hz_inverse, 0, z, args=(A, a, ok))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, A, a, ok): 
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(A, a, ok))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$A$",r"$\alpha$", r"$\Omega_k$"]

class NGCG(Likelihood):

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        A = params_values['A']
        a = params_values['a'] 
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, A, a, w)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, A, a, w) 

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, A, a, w): 
        Hz = np.sqrt(om*(1+z)**3 + ((1-om)*(1+z)**3)*(1-A*(1-(1+z)**(3*w*(1+a))))**(1/(1+a)))
        return 1.0 / Hz

    def distmod(self, om, A, a, w):
        zx = self.z_data
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, A, a, w))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, A, a, w): 
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, A, a, w))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"\Omega_m", r"$A$", r"$\alpha$", r"$w$"]

class DGP(Likelihood):

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        rc = params_values['rc']
        ok = params_values['ok']
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, rc, ok)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(rc, ok)

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, rc, ok):
        Hz = np.sqrt(ok*((1+z)**2)+ (((np.sqrt(((1 - ok - 2*(np.sqrt(rc)*np.sqrt(1-ok)))*((1+z)**3))+ rc )) + np.sqrt(rc) )**2))
        return 1.0 / Hz

    def distmod(self, rc, ok):
        zx = self.z_data
        x = np.array([quad(self.Hz_inverse, 0, z, args=(rc, ok))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, rc, ok):
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(rc, ok))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{rc}$", r"$\Omega_k$"]

class GAL(Likelihood): 

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

        # list of redshifts to interpolate between.
        self.z_interp = np.geomspace(0.0001, 2.5, 1000) 

        # Check if the covariance matrix is a zero matrix or not. (ie. Have we used fitopts?)
        # This is done to speed up the job and use a different likelihood function if not.
        if np.sum(self.cov_arr) == 0:
            self.cov = self.mu_error
            self.like_func = log_likelihood
            print('Covariance matrix is a zero matrix, check Fitops')
        else:
            self.like_func = cov_log_likelihood  


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        og = params_values['og']
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, og)
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, og)

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, og): 
        ok  = 1 - om - og
        Hz = np.sqrt(0.5*ok*(1+z)**2 + 0.5*om*(1+z)**3 + np.sqrt(og + 0.25*((om*(1+z)+ok)**2)*(1+z)**4))
        return 1.0 / Hz

    def distmod(self, om, og):
        zx = self.z_data
        ok = 1.0 - om - og
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, og))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, og):
        ok = 1.0 - om - og
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, og))[0] for z in zx])
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{g}$"]