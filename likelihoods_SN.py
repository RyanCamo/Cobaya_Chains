import numpy as np
from scipy.integrate import quad
import pandas as pd

# Calculates the likelihood for different models.
# Organised as:
## 1. Data to test against - Currently set up for DES5YR_UNBIN
## 2. The likelihood function
## 3. Models

#### Data 
## Options: - options not used - this is just for the DES5YR_UNBIN
## DES5YR UNBINNED = DES5YR_UNBIN
## DES5YR BINNED = DES5YR_BIN 
## DES3YR UNBINNED = DES3YR_UNBIN
## DES3YR BINNED = DES3YR_BIN

# Each data set needs to be imported differently

# Current data being used:
DataToUse = 'DES5YR_UNBIN'
DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
zs = DES5YR_UNBIN['zCMB']
mu = DES5YR_UNBIN['MU']
error = DES5YR_UNBIN['MUERR']
cov_arr = np.genfromtxt("/Users/RyanCamo/Desktop/Cobaya/data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
cov1 = cov_arr.reshape(1867,1867) 
mu_diag = np.diag(error)**2
cov = mu_diag+cov1

#### The SN-Likelihood function:

def cov_log_likelihood(mu_model, mu, cov):
    delta = np.array([mu_model - mu])
    inv_cov = np.linalg.inv(cov)
    deltaT = np.transpose(delta)
    chit2 = np.sum(delta @ inv_cov @ deltaT)
    B = np.sum(delta @ inv_cov)
    C = np.sum(inv_cov)
    chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
    return -0.5*chi2 

#### Non-Standard Models:

# 1) Flat Cosmological Constant with 1x paramater, \Omega_M - DONE
def FLCDM_Hz_inverse(z,om, ol):
    #ol = 1-om
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz
    
def FLCDM(om):
    ol = 1 - om
    x = np.array([quad(FLCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs) 
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp


# 2) Cosmological Constant with 2x paramaters, \Omega_M and \Omega_{\Lambda} - DONE
def LCDM_Hz_inverse(z,om,ol):
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz

def LCDM(om, ol):
    ok = 1.0 - om - ol
    x = np.array([quad(LCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zs])
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
    label = ["$\Omega_m$","$\Omega_{\Lambda}$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp


# 3) Flat Constant wCDM with 2x paramaters, \Omega_M and \omega - DONE
def FwCDM_Hz_inverse(z,om,w):
    ol = 1 - om
    Hz = np.sqrt((om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return 1.0 / Hz

def FwCDM(om, w):
    ol = 1 - om
    x = np.array([quad(FwCDM_Hz_inverse, 0, z, args=(om, w))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$\omega$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 4) Constant wCDM with 3x parameters, \Omega_M, \Omega_{\Lambda} and \omega - DONE
def wCDM_Hz_inverse(z,om,ol,w):
    omega_k = 1.0 - om - ol
    Hz = np.sqrt((omega_k*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return 1.0 / (Hz)
    
def wCDM(om,ol,w):
    ok = 1.0 - om - ol
    x = np.array([quad(wCDM_Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zs])
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
    label = ["$\Omega_m$","$\Omega_{\Lambda}$",r"$\omega$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 5) Flat w(a) with 3x parameters, \Omega_M, \omega_0 and \omega_a - DONE
def Fwa_Hz_inverse(z,om,w0,wa):
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0+wa))) * (np.exp(-3*wa*(1-((1+z)**(-1))))) ) )
    return 1.0 / Hz

def Fwa(om, w0, wa):
    x = np.array([quad(Fwa_Hz_inverse, 0, z, args=(om, w0, wa))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$w_0$","$w_a$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 15) IDE1 Q = H e rho_x
def IDE_Hz_inverse1(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(cdm*(1+z)**3 + ol*(1-(e/(3*w + e)))*(1+z)**((3*(1+w)+e)) + ok*(1+z)**2) 
    return 1.0 / Hz

def IDE1(cdm,ol,w,e):
    ok = 1 -ol - cdm
    x = np.array([quad(IDE_Hz_inverse1, 0, z, args=(cdm, ol, w, e))[0] for z in zs])
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
    label = [r"$\Omega_{CDM}$", r"$\Omega_{DE}$", r"$\omega$", r"$\epsilon$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 16) IDE2 Q = H e rho_c
def IDE_Hz_inverse2(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(cdm*((1+z)**(3-e)) * (1-(e)/(3*w +e )) + ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
    return 1.0 / Hz

def IDE2(cdm,ol,w,e):
    ok = 1 -ol - cdm
    x = np.array([quad(IDE_Hz_inverse2, 0, z, args=(cdm, ol, w, e))[0] for z in zs])
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
    label = [r"$\Omega_{CDM}$", r"$\Omega_{DE}$", r"$\omega$", r"$\epsilon$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 18) IDE4 Q = H e [rho_c * rho_x / (rho_c + rho_x)]
def IDE_Hz_inverse4(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    IDE4_const = ( ((1+z)**(-1*(e+(3*w)))) + (ol/cdm) )**(-e/(e+(3*w)))
    Hz = np.sqrt( (cdm*IDE4_const*(1+z)**(3-e)) + IDE4_const*ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
    return 1.0 / Hz

def IDE4(cdm,ol,w,e):
    ok = 1 -ol - cdm
    x = np.array([quad(IDE_Hz_inverse4, 0, z, args=(cdm, ol, w, e))[0] for z in zs])
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
    label = [r"$\Omega_{CDM}$", r"$\Omega_{DE}$", r"$\omega$", r"$\epsilon$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 19) Flat w(z) with 3x parameters, \Omega_M, \omega_0 and \omega_a - DONE
def Fwz_Hz_inverse(z,om,w0,wz):
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0-wz))) * (np.exp(3*wz*z)) ) )
    return 1.0 / Hz

def Fwz(om,w0,wz):
    x = np.array([quad(Fwa_Hz_inverse, 0, z, args=(om, w0, wz))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$w_0$","$w_z$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 6) Cardassian with 3x parameters, \Omega_M, q and n
def FCa_Hz_inverse(z, om, q ,n ):
    Hz = np.sqrt(
        (om*((z+1)**3))*(1+(((om**(-q))-1)*((z+1)**(3*q*(n-1)))))**(1/q))
    return 1.0 / Hz

def FCa(om, q, n):
    x = np.array([quad(FCa_Hz_inverse, 0, z, args=(om, q, n))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$q$","$n$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 9) Flat General Chaplygin 2x parameters, A and \alpha
def FGChap_Hz_inverse(z, A, a):
    Hz = np.sqrt((A + (1-A)*((1+z)**(3*(1+a))))**(1.0/(1+a)))
    return 1.0 / Hz

def FGChap(A, a):
    x = np.array([quad(FGChap_Hz_inverse, 0, z, args=(A, a))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$A$", r"$\alpha$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 8) Chaplygin 2x parameters, A and \Omega_K
def Chap_Hz_inverse(z, A, ok):
    Hz = np.sqrt(ok*((1+z)**2)+(1-ok)*np.sqrt(A + (1-A)*((1+z)**6)))
    return 1.0 / Hz

def Chap(A, ok):
    x = np.array([quad(Chap_Hz_inverse, 0, z, args=(A, ok))[0] for z in zs])
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
    label = ["$A$","$\Omega_k$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp


# 10) General Chaplygin 3x parameters, \Omega_K, A and \alpha
def GChap_Hz_inverse(z, ok, A ,a):
    Hz = np.sqrt((ok*((1+z)**2)) + (1-ok)*(A + (1-A)*((1+z)**(3*(1+a))))**(1/(1+a)))
    return 1.0 / Hz

def GChap(A, a, ok):
    x = np.array([quad(GChap_Hz_inverse, 0, z, args=(ok, A, a))[0] for z in zs])
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
    label = ["$\Omega_K$","$A$",r"$\alpha$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 11) DGP 2x parameters, \Omega_rc, and \Omega_K
def DGP_Hz_inverse(z, rc, ok):
    Hz = np.sqrt(ok*((1+z)**2)+ (((np.sqrt(((1 - ok - 2*(np.sqrt(rc)*np.sqrt(1-ok)))*((1+z)**3))+ rc )) + np.sqrt(rc) )**2)) 
    return 1.0 / Hz

def DGP(rc, ok):
    x = np.array([quad(DGP_Hz_inverse, 0, z, args=(rc, ok))[0] for z in zs])
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
    label = [r"$\Omega_{rc}$", r"$\Omega_K$"]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 22) New General Chaplygin 4x parameters, \Omega_K, A and \alpha, w
def NGCG_Hz_inverse(z, om, A ,a, w):
    Hz = np.sqrt(om*(1+z)**3 + ((1-om)*(1+z)**3)*(1-A*(1-(1+z)**(3*w*(1+a))))**(1/(1+a)))
    return 1.0 / Hz

def NGCG(om, A, a, w):
    x = np.array([quad(NGCG_Hz_inverse, 0, z, args=(om, A, a, w))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["\Omega_m", "$A$",r"$\alpha$","$\Omega_K$",]
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

# 11) Galileon Tracker Solution 2x parameters, \Omega_m, \Omega_g
def GAL_Hz_inverse(z, om, og):
    ok  = 1 - om - og
    Hz = np.sqrt(0.5*ok*(1+z)**2 + 0.5*om*(1+z)**3 + np.sqrt(og + 0.25*((om*(1+z)+ok)**2)*(1+z)**4))
    return 1.0 / Hz

def GAL(om, og):
    ok  = 1 - om - og
    x = np.array([quad(GAL_Hz_inverse, 0, z, args=(om, og))[0] for z in zs])
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
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

if __name__ == "__main__":
    #logp = LCDM(0.31,0.7)
    logp = FwCDM(0.3,-1)
    print(logp)
