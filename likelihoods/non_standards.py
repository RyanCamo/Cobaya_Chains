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

##############################################################################################################################################

class NAME(Likelihood): ################### CHANGE NAME

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
        om = params_values['om']  ############ CHANGE PARAMS
        ol = params_values['ol']
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol, w) ####### CHANGE PARAMS
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol, w) ####### CHANGE PARAMS

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        Hz = np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w)))) ####### CHANGE FUNC
        return 1.0 / Hz

    def distmod(self, om, ol, w): ####### CHANGE PARAMS
        zx = self.z_data
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)            ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)  ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$", r"$w$"] ####### CHANGE PARAMS

class NAME(Likelihood): ################### CHANGE NAME

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
        om = params_values['om']  ############ CHANGE PARAMS
        ol = params_values['ol']
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol, w) ####### CHANGE PARAMS
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol, w) ####### CHANGE PARAMS

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        Hz = np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w)))) ####### CHANGE FUNC
        return 1.0 / Hz

    def distmod(self, om, ol, w): ####### CHANGE PARAMS
        zx = self.z_data
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)            ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)  ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$", r"$w$"] ####### CHANGE PARAMS

class NAME(Likelihood): ################### CHANGE NAME

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
        om = params_values['om']  ############ CHANGE PARAMS
        ol = params_values['ol']
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol, w) ####### CHANGE PARAMS
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol, w) ####### CHANGE PARAMS

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        Hz = np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w)))) ####### CHANGE FUNC
        return 1.0 / Hz

    def distmod(self, om, ol, w): ####### CHANGE PARAMS
        zx = self.z_data
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)            ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)  ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$", r"$w$"] ####### CHANGE PARAMS

class NAME(Likelihood): ################### CHANGE NAME

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
        om = params_values['om']  ############ CHANGE PARAMS
        ol = params_values['ol']
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol, w) ####### CHANGE PARAMS
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol, w) ####### CHANGE PARAMS

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        Hz = np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w)))) ####### CHANGE FUNC
        return 1.0 / Hz

    def distmod(self, om, ol, w): ####### CHANGE PARAMS
        zx = self.z_data
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)            ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)  ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$", r"$w$"] ####### CHANGE PARAMS

class NAME(Likelihood): ################### CHANGE NAME

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
        om = params_values['om']  ############ CHANGE PARAMS
        ol = params_values['ol']
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol, w) ####### CHANGE PARAMS
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol, w) ####### CHANGE PARAMS

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        Hz = np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w)))) ####### CHANGE FUNC
        return 1.0 / Hz

    def distmod(self, om, ol, w): ####### CHANGE PARAMS
        zx = self.z_data
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)            ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)  ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$", r"$w$"] ####### CHANGE PARAMS

class NAME(Likelihood): ################### CHANGE NAME

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
        om = params_values['om']  ############ CHANGE PARAMS
        ol = params_values['ol']
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol, w) ####### CHANGE PARAMS
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol, w) ####### CHANGE PARAMS

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        Hz = np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w)))) ####### CHANGE FUNC
        return 1.0 / Hz

    def distmod(self, om, ol, w): ####### CHANGE PARAMS
        zx = self.z_data
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)            ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)  ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$", r"$w$"] ####### CHANGE PARAMS

class NAME(Likelihood): ################### CHANGE NAME

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
        om = params_values['om']  ############ CHANGE PARAMS
        ol = params_values['ol']
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol, w) ####### CHANGE PARAMS
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol, w) ####### CHANGE PARAMS

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        Hz = np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w)))) ####### CHANGE FUNC
        return 1.0 / Hz

    def distmod(self, om, ol, w): ####### CHANGE PARAMS
        zx = self.z_data
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)            ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)  ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$", r"$w$"] ####### CHANGE PARAMS

class NAME(Likelihood): ################### CHANGE NAME

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
        om = params_values['om']  ############ CHANGE PARAMS
        ol = params_values['ol']
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol, w) ####### CHANGE PARAMS
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol, w) ####### CHANGE PARAMS

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        Hz = np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w)))) ####### CHANGE FUNC
        return 1.0 / Hz

    def distmod(self, om, ol, w): ####### CHANGE PARAMS
        zx = self.z_data
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)            ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)  ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$", r"$w$"] ####### CHANGE PARAMS

class NAME(Likelihood): ################### CHANGE NAME

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
        om = params_values['om']  ############ CHANGE PARAMS
        ol = params_values['ol']
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol, w) ####### CHANGE PARAMS
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol, w) ####### CHANGE PARAMS

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        Hz = np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w)))) ####### CHANGE FUNC
        return 1.0 / Hz

    def distmod(self, om, ol, w): ####### CHANGE PARAMS
        zx = self.z_data
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)            ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)  ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$", r"$w$"] ####### CHANGE PARAMS

class NAME(Likelihood): ################### CHANGE NAME

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
        om = params_values['om']  ############ CHANGE PARAMS
        ol = params_values['ol']
        w = params_values['w'] 
        # interpolates by default. Can be changed using the interp flag in the input .yaml
        if self.interp == True:       
            dl_interp = interp_dl(self.lum_dist_interp, om, ol, w) ####### CHANGE PARAMS
            dl_data = dl_interp(self.z_data)
            dist_mod = 5 * np.log10(dl_data)
        elif self.interp == False:
            dist_mod = self.distmod(om, ol, w) ####### CHANGE PARAMS

        return self.like_func(dist_mod, self.mu_data, self.cov)

    def Hz_inverse(self, z, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        Hz = np.sqrt((ok*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w)))) ####### CHANGE FUNC
        return 1.0 / Hz

    def distmod(self, om, ol, w): ####### CHANGE PARAMS
        zx = self.z_data
        ok = 1.0 - om - ol ####### CHANGE PARAMS
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)            ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def lum_dist_interp(self, om, ol, w): ####### CHANGE PARAMS
        ok = 1.0 - om - ol
        zx = self.z_interp
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zx]) ####### CHANGE PARAMS
        if ok < 0.0:
            R0 = 1 / np.sqrt(-ok)
            D = R0 * np.sin(x / R0)  ####### CHECK CURVATURE?
        elif ok > 0.0:
            R0 = 1 / np.sqrt(ok)
            D = R0 * np.sinh(x / R0)
        else:
            D = x
        lum_dist = D * (1 + zx) 
        return lum_dist

    def label(self):
        return [r"$\Omega_{\text{m}}$", r"$\Omega_{\Lambda}$", r"$w$"] ####### CHANGE PARAMS