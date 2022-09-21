from tkinter import W
from cobaya.likelihood import Likelihood
import numpy as np
from scipy.integrate import quad

class FLCDM(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) 
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    #def get_requirements(self):
    #    """
    #     return dictionary specifying quantities calculated by a theory code are needed
    #
    #     e.g. here we need C_L^{tt} to lmax=2500 and the H0 value
    #    """
    #    return {'Cl': {'tt': 2500}, 'H0': None}

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        mu_model = self.distmod(om)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2

    def Hz_inverse(self, z, om, ol):
        Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
        return 1.0 / Hz

    def distmod(self, om):
        ol = 1 - om
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, ol))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod 

    def label(self):
        return [r"$\Omega_m$"]


class LCDM(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        ol = params_values['ol']
        mu_model = self.distmod(om, ol)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2


    def Hz_inverse(self, z, om, ol):
        Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
        return 1.0 / Hz

    def distmod(self, om, ol):
        zx = self.zs
        ok = 1 - om - ol
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

    def label(self):
        return [r"$\Omega_m$",r"$\Omega_{\Lambda}$"]

class FwCDM(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        w = params_values['w']
        mu_model = self.distmod(om, w)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2


    def Hz_inverse(self, z, om, w):
        ol = 1 - om
        Hz = np.sqrt((om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
        return 1.0 / Hz

    def distmod(self, om, w):
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, w))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod

    def label(self):
        return [r"$\Omega_m$",r"$\omega$"]


class wCDM(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        ol = params_values['ol']
        w = params_values['w']
        mu_model = self.distmod(om, ol, w)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2


    def Hz_inverse(self, z, om, ol, w):
        omega_k = 1.0 - om - ol
        Hz = np.sqrt((omega_k*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
        return 1.0 / Hz

    def distmod(self, om, ol, w):
        ok = 1.0 - om - ol
        zx = self.zs
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

    def label(self):
        return [r"$\Omega_m$",r"$\Omega_{\Lambda}$",r"$\omega$"]

class Fwa(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        w0 = params_values['w0']
        wa = params_values['wa']
        mu_model = self.distmod(om, w0, wa)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2


    def Hz_inverse(self, z, om, w0, wa):
        ol = 1 - om
        Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0+wa))) * (np.exp(-3*wa*(1-((1+z)**(-1))))) ) )
        return 1.0 / Hz

    def distmod(self, om, w0, wa):
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, w0, wa))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod

    def label(self):
        return [r"$\Omega_m$",r"$w_0$",r"$w_a$"]


class Fwz(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        w0 = params_values['w0']
        wz = params_values['wz']
        mu_model = self.distmod(om, w0, wz)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2


    def Hz_inverse(self, z, om, w0, wz):
        ol = 1 - om 
        Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0-wz))) * (np.exp(3*wz*z)) ) )
        return 1.0 / Hz

    def distmod(self, om, w0, wz):
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, w0, wz))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod

    def label(self):
        return [r"$\Omega_m$",r"$w_0$",r"$w_z$"]


class FCa(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        q = params_values['q']
        n = params_values['n']
        mu_model = self.distmod(om, q, n)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2


    def Hz_inverse(self, z, om, q, n):
        Hz = np.sqrt((om*((z+1)**3))*(1+(((om**(-q))-1)*((z+1)**(3*q*(n-1)))))**(1/q))
        return 1.0 / Hz

    def distmod(self, om, q, n):
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, q, n))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod

    def label(self):
        return [r"$\Omega_m$",r"$q$","$n$"]


class FGChap(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        A = params_values['A']
        a = params_values['q']
        mu_model = self.distmod(A, a)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2


    def Hz_inverse(self, z, A, a):
        Hz = np.sqrt((A + (1-A)*((1+z)**(3*(1+a))))**(1.0/(1+a)))
        return 1.0 / Hz

    def distmod(self, A, a):
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(A, a))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod

    def label(self):
        return [r"$A$", r"$\alpha$"]

class DGP(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        rc = params_values['rc']
        ok = params_values['ok']
        mu_model = self.distmod(rc, ok)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2

    def Hz_inverse(self, z, rc, ok):
        Hz = np.sqrt(ok*((1+z)**2)+ (((np.sqrt(((1 - ok - 2*(np.sqrt(rc)*np.sqrt(1-ok)))*((1+z)**3))+ rc )) + np.sqrt(rc) )**2)) 
        return 1.0 / Hz

    def distmod(self, rc, ok):
        zx = self.zs
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

    def label(self):
        return [r"$\Omega_{rc}$", r"$\Omega_K$"]


class Chap(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        A = params_values['A']
        ok = params_values['ok']
        mu_model = self.distmod(A, ok)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2

    def Hz_inverse(self, z, A, ok):
        Hz = np.sqrt(ok*((1+z)**2)+(1-ok)*np.sqrt(A + (1-A)*((1+z)**6)))
        return 1.0 / Hz

    def distmod(self, A, ok):
        zx = self.zs
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

    def label(self):
        return [r"$A$",r"$\Omega_k$"]

class GChap(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        A = params_values['A']
        ok = params_values['ok']
        a = params_values['a']
        mu_model = self.distmod(ok, A, a)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2

    def Hz_inverse(self, z, ok, A, a):
        Hz = np.sqrt((ok*((1+z)**2)) + (1-ok)*(A + (1-A)*((1+z)**(3*(1+a))))**(1/(1+a)))
        return 1.0 / Hz

    def distmod(self, ok, A, a):
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(ok, A, a))[0] for z in zx])
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

    def label(self):
        return [r"$\Omega_K$", r"$A$",r"$\alpha$"]

class NGCG(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

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
        mu_model = self.distmod(om, A, a, w)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2

    def Hz_inverse(self, z, om, A, a, w):
        Hz = np.sqrt(om*(1+z)**3 + ((1-om)*(1+z)**3)*(1-A*(1-(1+z)**(3*w*(1+a))))**(1/(1+a)))
        return 1.0 / Hz

    def distmod(self, om, A, a, w):
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(om, A, a, w))[0] for z in zx])
        D = x
        lum_dist = D * (1 + zx) 
        dist_mod = 5 * np.log10(lum_dist)
        return dist_mod

    def label(self):
        return [r"$\Omega_m$", r"$A$",r"$\alpha$",r"$\omega$"]


class GAL(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        om = params_values['om']
        og = params_values['og']
        mu_model = self.distmod(om, og)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2

    def Hz_inverse(self, z, om, og):
        ok  = 1 - om - og
        Hz = np.sqrt(0.5*ok*(1+z)**2 + 0.5*om*(1+z)**3 + np.sqrt(og + 0.25*((om*(1+z)+ok)**2)*(1+z)**4))
        return 1.0 / Hz

    def distmod(self, om, og):
        ok = 1 - om - og
        zx = self.zs
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

    def label(self):
        return [r"$\Omega_m$", r"$\Omega_g$" ]


class IDE1(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        cdm = params_values['cdm']
        ol = params_values['ol']
        w = params_values['w']
        e = params_values['e']
        mu_model = self.distmod(cdm, ol, w, e)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2

    def Hz_inverse(self, z, cdm, ol, w, e):
        ok = 1.0 - cdm - ol
        Hz = np.sqrt(cdm*(1+z)**3 + ol*(1-(e/(3*w + e)))*(1+z)**((3*(1+w)+e)) + ok*(1+z)**2) 
        return 1.0 / Hz

    def distmod(self, cdm, ol, w, e):
        ok = 1 - ol - cdm
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(cdm, ol, w, e))[0] for z in zx])
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

    def label(self):
        return [r"$\Omega_{m}$", r"$\Omega_{x}$", r"$\omega$", r"$\varepsilon$"]


class IDE2(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        cdm = params_values['cdm']
        ol = params_values['ol']
        w = params_values['w']
        e = params_values['e']
        mu_model = self.distmod(cdm, ol, w, e)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2

    def Hz_inverse(self, z, cdm, ol, w, e):
        ok = 1.0 - cdm - ol
        Hz = np.sqrt(cdm*((1+z)**(3-e)) * (1-(e)/(3*w +e )) + ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
        return 1.0 / Hz

    def distmod(self, cdm, ol, w, e):
        ok = 1 - ol - cdm
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(cdm, ol, w, e))[0] for z in zx])
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

    def label(self):
        return [r"$\Omega_{m}$", r"$\Omega_{x}$", r"$\omega$", r"$\varepsilon$"]

class IDE3(Likelihood):

    def initialize(self):
        """
         Prepare any computation, importing any necessary code, files, etc.

         e.g. here we load some data file, with default cl_file set in .yaml below,
         or overridden when running Cobaya.
        """
        # Current data being used:
        DataToUse = 'DES5YR_UNBIN'
        DES5YR_UNBIN = np.genfromtxt(self.Data, names=True)
        #DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
        self.zs = DES5YR_UNBIN['zCMB']
        self.mu = DES5YR_UNBIN['MU']
        self.error = DES5YR_UNBIN['MUERR']
        #cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
        cov_arr = np.genfromtxt(self.Cov, comments='#',dtype=None)
        cov1 = cov_arr.reshape(1867,1867) # This needs to be better generalised to other datasets.
        mu_diag = np.diag(self.error)**2
        self.cov = mu_diag+cov1

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        cdm = params_values['cdm']
        ol = params_values['ol']
        w = params_values['w']
        e = params_values['e']
        mu_model = self.distmod(cdm, ol, w, e)
        delta = np.array([mu_model - self.mu])
        inv_cov = np.linalg.inv(self.cov)
        deltaT = np.transpose(delta)
        chit2 = np.sum(delta @ inv_cov @ deltaT)
        B = np.sum(delta @ inv_cov)
        C = np.sum(inv_cov)
        chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
        return -chi2 / 2

    def Hz_inverse(self, z, cdm, ol, w, e):
        ok = 1.0 - cdm - ol
        IDE4_const = ( ((1+z)**(-1*(e+(3*w)))) + (ol/cdm) )**(-e/(e+(3*w)))
        Hz = np.sqrt( (cdm*IDE4_const*(1+z)**(3-e)) + IDE4_const*ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
        return 1.0 / Hz

    def distmod(self, cdm, ol, w, e):
        ok = 1 - ol - cdm
        zx = self.zs
        x = np.array([quad(self.Hz_inverse, 0, z, args=(cdm, ol, w, e))[0] for z in zx])
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

    def label(self):
        return [r"$\Omega_{m}$", r"$\Omega_{x}$", r"$\omega$", r"$\varepsilon$"]





    