from cobaya.likelihood import Likelihood
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from functools import lru_cache
from data.CMB_BAO.imports import get_CMB_BAO_data

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

# CMB/BAO
# Using for the uncorrelated data
def CMB_BAO_log_likelihood(f, f_err, model): 
    delta = model - f
    chit2 = np.sum(delta**2 / f_err**2)
    chi2 = chit2 
    return -0.5*chi2

# For the correlated data
def CMB_BAO_cov_log_likelihood(mu_model, mu, cov):
    delta = mu_model - mu
    inv_cov = np.linalg.inv(cov)
    deltaT = np.transpose(delta)
    chit2 = np.sum(delta @ inv_cov @ deltaT)
    chi2 = chit2
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
        # Load in SN data
        if (self.HD_path != False) & (self.cov_path != False):
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

        if self.CMB_BAO != False:
            self.BOSS_cov, self.BOSS_zz, self.BOSS_data, self.eBOSS_LRG_cov, self.eBOSS_LRG_zz, self.eBOSS_LRG_data, self.eBOSS_QSO_cov, self.eBOSS_QSO_zz, self.eBOSS_QSO_data, self.DMonDH_zz, self.DMonDH_data, self.DMonDH_err, self.DMonDM_zz, self.DMonDM_data, self.DMonDM_err, self.DMonDV_zz, self.DMonDV_data, self.DMonDV_err = get_CMB_BAO_data()
            


    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        SN_like = 0
        CMB_BAO_like = 0
    
        if (self.HD_path != False) & (self.cov_path != False):
            # interpolates by default. Can be changed using the interp flag in the input .yaml
            if self.interp == True:       
                dl_interp = interp_dl(self.lum_dist_interp, om)
                dl_data = dl_interp(self.z_data)
                dist_mod = 5 * np.log10(dl_data)
            elif self.interp == False:
                dist_mod = self.distmod(om)      
            SN_like = self.like_func(dist_mod, self.mu_data, self.cov)

        if self.CMB_BAO != False:
            # uncorrelated data
            DM_on_DV_model = self.DM_on_DV(self.DMonDV_zz, om)
            DM_on_DM_model = self.DM_on_DM(self.DMonDM_zz, om)
            DM_on_DH_model = self.DM_on_DH(self.DMonDH_zz, om)
            like_DM_on_DV = CMB_BAO_log_likelihood(self.DMonDV_data, self.DMonDV_err, DM_on_DV_model)
            like_DM_on_DM = CMB_BAO_log_likelihood(self.DMonDM_data, self.DMonDM_err, DM_on_DM_model)
            like_DM_on_DH = CMB_BAO_log_likelihood(self.DMonDH_data, self.DMonDH_err, DM_on_DH_model)

            # correlated data
            BOSS_DM_on_DM = self.DM_on_DM(self.BOSS_zz, om)
            BOSS_DM_on_DH = self.DM_on_DH(self.BOSS_zz, om)
            BOSS_model = np.array([elem for singleList in list(zip(BOSS_DM_on_DM, BOSS_DM_on_DH)) for elem in singleList])
            like_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, self.BOSS_data, self.BOSS_cov) 

            eBOSS_LRG_DM_on_DM = self.DM_on_DM(self.eBOSS_LRG_zz, om)
            eBOSS_LRG_DM_on_DH = self.DM_on_DH(self.eBOSS_LRG_zz, om)
            eBOSS_LRG_model = np.array([elem for singleList in list(zip(eBOSS_LRG_DM_on_DM, eBOSS_LRG_DM_on_DH)) for elem in singleList])
            like_LRG = CMB_BAO_cov_log_likelihood(eBOSS_LRG_model, self.eBOSS_LRG_data, self.eBOSS_LRG_cov)

            eBOSS_QSO_DM_on_DM = self.DM_on_DM(self.eBOSS_QSO_zz, om)
            eBOSS_QSO_DM_on_DH = self.DM_on_DH(self.eBOSS_QSO_zz, om)
            eBOSS_QSO_model = np.array([elem for singleList in list(zip(eBOSS_QSO_DM_on_DM, eBOSS_QSO_DM_on_DH)) for elem in singleList])
            like_QSO = CMB_BAO_cov_log_likelihood(eBOSS_QSO_model, self.eBOSS_QSO_data, self.eBOSS_QSO_cov)
            CMB_BAO_like = like_DM_on_DH + like_DM_on_DM + like_DM_on_DV  #+ like_BOSS + like_LRG + like_QSO

        return SN_like + CMB_BAO_like

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

    def DM_on_DV(self, z_, om):
        ol = 1 - om
        last_scat = np.array([quad(self.Hz_inverse, 0, 1090, args=(om, ol))[0]]) # Last Scattering
        ang_star = last_scat / (1+1090)
        DV_rd_model = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol))[0] for z in z_]) # dv/rd data
        ang_dist = DV_rd_model / (1 + z_)
        Hz = 1 / self.Hz_inverse(z_, om, ol)
        DV = ((1 + z_)**2 * ang_dist**2 * (z_)/Hz)**(1/3)
        model = (ang_star)*(1+1090) / DV
        return model

    def DM_on_DM(self, z_, om):
        ol = 1 - om
        m = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol))[0] for z in z_]) #dm/rd data
        last_scat = np.array([quad(self.Hz_inverse, 0, 1090, args=(om, ol))[0]]) # Last Scattering
        ang_star_0 = last_scat / (1+1090)
        ang_dist_1 =  m / (1 + z_)
        model = ((ang_star_0)*(1+1090)) / ((ang_dist_1)*(1+z_))
        return model  

    def DM_on_DH(self, z_, om):
        ol = 1 - om
        Hz = 1 / self.Hz_inverse(z_, om, ol)
        last_scat = np.array([quad(self.Hz_inverse, 0, 1090, args=(om, ol))[0]]) # Last Scattering
        ang_star = last_scat / (1+1090)
        model = ((ang_star)*(1+1090)) / Hz
        return model 

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

if __name__ == "__main__":
    from chainconsumer import ChainConsumer
    c = ChainConsumer()
    SN = np.loadtxt('chains/Test_Class.1.txt', usecols=(2), comments='#')
    SN_weights = np.loadtxt('chains/Test_Class.1.txt', usecols=(0), comments='#')
    c.add_chain(SN, weights=SN_weights, name='SN')
    print(c.analysis.get_summary(chains="SN"))