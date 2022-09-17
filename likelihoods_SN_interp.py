from cmath import log10
from re import M
import numpy as np
from scipy.integrate import quad
import pandas as pd
import sys
from pathlib import Path
from scipy.interpolate import interp1d
from functools import lru_cache
import time
#start_time = time.time()

# Calculates the likelihood for different models.
# Organised as:
## 1. Data to test against - Currently set up for the mock data
## 2. The likelihood function
## 3. Models

# Mock Data
cov_path = Path("data/data_TESTS/cov_51.txt")
HD_path = Path("data/data_TESTS/HD_51.txt")
arr_size = int(np.genfromtxt(cov_path, comments='#',dtype=None)[0])
DES5YR_UNBIN = np.genfromtxt(HD_path, names=True, comments='#')
cov_arr = np.genfromtxt(cov_path, comments='#',dtype=None)[1:]

z_data = DES5YR_UNBIN['zCMB']
mu = DES5YR_UNBIN['MU']
error = DES5YR_UNBIN['MUERR']
cov1 = cov_arr.reshape(arr_size,arr_size) 
mu_diag = np.diag(error)**2
cov = mu_diag+cov1

zs = np.geomspace(0.0001, 2.5, 1000) # list of redshifts to interpolate between.

@lru_cache(maxsize=4096)
def interp_dl(model, *params):
    z_interp = np.geomspace(0.0001, 2.5, 1000)
    dl_interp = interp1d(z_interp, model(*params), kind="linear")
    return dl_interp


def getH0(dist_mod, mu):
    print('This function is not working yet...')
    exit()
    print(mu-dist_mod)
    offset = (np.sum(mu-dist_mod)/len(z_data))
    offset_standardized = offset + 5*log10(c) 
    avg = (np.sum(mu)/len(z_data))
    print(offset)
    print(5*log10(c/offset))
    #h0_= 1/((10**((abs(offset-avg))/5))/c)
    h0_= c * 10**(offset_standardized/5)
    #h0_= 1/((10**((abs(offset-dist_mod[0]))/5))/c)
    return h0_

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
c = 299792458

def FLCDM(om):
    dl_interp = interp_dl(dl_FLCDM, om)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def LCDM(om, ol):
    dl_interp = interp_dl(dl_LCDM, om, ol)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def FwCDM(om, w):
    dl_interp = interp_dl(dl_FwCDM, om, w)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def wCDM(om, ol, w):
    dl_interp = interp_dl(dl_wCDM, om, ol, w)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def IDEA(cdm,w,e):
    dl_interp = interp_dl(dl_IDEA, cdm,w,e)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def IDEB(cdm, ol, w, e, ob):
    dl_interp = interp_dl(dl_IDEB, cdm, ol, w, e, ob)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def IDEC(cdm,ob,w,e):
    dl_interp = interp_dl(dl_IDEC, cdm,ob,w,e)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def Fwa(om, w0, wa):
    dl_interp = interp_dl(dl_wCDM, om, w0, wa)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def Fwz(om,w0,wz):
    dl_interp = interp_dl(dl_Fwz, om,w0,wz)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def FCa(om, q, n):
    dl_interp = interp_dl(dl_FCa, om, q, n)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def FGChap(A, a):
    dl_interp = interp_dl(dl_FGChap, A, a)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def Chap(A, ok):
    dl_interp = interp_dl(dl_Chap, A, ok)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def GChap(A, a, ok):
    dl_interp = interp_dl(dl_GChap, A, a, ok)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def DGP(rc, ok):
    dl_interp = interp_dl(dl_DGP, rc, ok)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def NGCG(om, A, a, w): 
    dl_interp = interp_dl(dl_NGCG, om, A, a, w)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

def GAL(om, og):
    dl_interp = interp_dl(dl_GAL, om, og)
    dl_data = dl_interp(z_data)
    dist_mod = 5 * np.log10(dl_data)
    logp = cov_log_likelihood(dist_mod, mu, cov)
    return logp

#########################################################

#### COMPUTES LUM_DIST FOR EACH MODEL.
# 1) Flat Cosmological Constant with 1x paramater, \Omega_M - DONE
def FLCDM_Hz_inverse(z,om, ol):
    #ol = 1-om
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz
    
def dl_FLCDM(om, H0=False):
    ol = 1 - om
    x = np.array([quad(FLCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs) 
    dist_mod = 5 * np.log10(lum_dist/10)
    if H0 == True:
        #h0_ = 1 / (FLCDM_Hz_inverse(0, om, ol))
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist


# 2) Cosmological Constant with 2x paramaters, \Omega_M and \Omega_{\Lambda} - DONE
def LCDM_Hz_inverse(z,om,ol):
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz

def dl_LCDM(om, ol, H0=False):
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
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist


# 3) Flat Constant wCDM with 2x paramaters, \Omega_M and \omega - DONE
def FwCDM_Hz_inverse(z,om,w):
    ol = 1 - om
    Hz = np.sqrt((om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return 1.0 / Hz

def dl_FwCDM(om, w, H0=False):
    ol = 1 - om
    x = np.array([quad(FwCDM_Hz_inverse, 0, z, args=(om, w))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 4) Constant wCDM with 3x parameters, \Omega_M, \Omega_{\Lambda} and \omega - DONE
def wCDM_Hz_inverse(z,om,ol,w):
    omega_k = 1.0 - om - ol
    Hz = np.sqrt((omega_k*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return 1.0 / (Hz)
    
def dl_wCDM(om,ol,w, H0=False):
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
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 5) Flat w(a) with 3x parameters, \Omega_M, \omega_0 and \omega_a - DONE
def Fwa_Hz_inverse(z,om,w0,wa):
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0+wa))) * (np.exp(-3*wa*(1-((1+z)**(-1))))) ) )
    return 1.0 / Hz

def dl_Fwa(om, w0, wa, H0=False):
    x = np.array([quad(Fwa_Hz_inverse, 0, z, args=(om, w0, wa))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 15) IDE1 Q = H e rho_x
#def IDE_Hz_inverse1(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(cdm*(1+z)**3 + ol*(1-(e/(3*w + e)))*(1+z)**((3*(1+w)+e)) + ok*(1+z)**2) 
    return 1.0 / Hz

#def dl_IDE1(cdm,ol,w,e, H0=False):
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
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 16) IDE2 Q = H e rho_c
#def IDE_Hz_inverse2(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(cdm*((1+z)**(3-e)) * (1-(e)/(3*w +e )) + ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
    return 1.0 / Hz

#def dl_IDE2(cdm,ol,w,e, H0=False):
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
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 18) IDE4 Q = H e [rho_c * rho_x / (rho_c + rho_x)]
#def IDE_Hz_inverse4(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    IDE4_const = ( ((1+z)**(-1*(e+(3*w)))) + (ol/cdm) )**(-e/(e+(3*w)))
    Hz = np.sqrt( (cdm*IDE4_const*(1+z)**(3-e)) + IDE4_const*ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
    return 1.0 / Hz

#def dl_IDE4(cdm,ol,w,e, H0=False):
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
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 15) IDE1 Q = H e rho_x
def IDE_Hz_inverseA(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(cdm*(1+z)**3 + ol*( ((e)/(w+e))*(1+z)**3 + ((w)/(w+e))*(1+z)**(3*(1+w+e))  )) 
    return 1.0 / Hz

def dl_IDEA(cdm,w,e, H0=False):
    ol = 1 - cdm
    x = np.array([quad(IDE_Hz_inverseA, 0, z, args=(cdm, ol, w, e))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 16) IDE2 Q = H e rho_c
def IDE_Hz_inverseB(z, cdm, ol, w, e, ob):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(ob*(1+z)**(3)+ ol*(1+z)**(3*(1+w)) + cdm*(((e)/(w+e))*(1+z)**(3*(1+w))  + ((w)/(w+e))*(1+z)**(3*(1-e)))) 
    return 1.0 / Hz

def dl_IDEB(cdm,ob,w,e, H0=False):
    ol = 1 -ob - cdm
    x = np.array([quad(IDE_Hz_inverseB, 0, z, args=(cdm, ol, w, e, ob))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 16) IDE2 Q = H e rho_c
#def IDE_Hz_inverseB_2(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(ol*(1+z)**(3*(1+w)) + cdm*(((e)/(w+e))*(1+z)**(3*(1+w))  + ((w)/(w+e))*(1+z)**(3*(1-e)))) 
    return 1.0 / Hz

#def dl_IDEB_2(cdm,w,e, H0=False):
    ol = 1 - cdm
    x = np.array([quad(IDE_Hz_inverseB_2, 0, z, args=(cdm, ol, w, e))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 18) IDE4 Q = H e [rho_c * rho_x / (rho_c + rho_x)]
#def IDE_Hz_inverseC_2(z, cdm, ol, w, e):
    constC = ((cdm)/(ol+cdm) + ((ol)/(ol+cdm))*(1+z)**(3*(w+e)))**(-(e)/(w+e))
    Hz = np.sqrt( cdm*constC*(1+z)**3 +  ol*constC*(1+z)**(3*(1+w+e))) 
    return 1.0 / Hz

#def dl_IDEC_2(cdm,w,e, H0=False):
    ol = 1 - cdm 
    x = np.array([quad(IDE_Hz_inverseC_2, 0, z, args=(cdm, ol, w, e))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 18) IDE4 Q = H e [rho_c * rho_x / (rho_c + rho_x)]
def IDE_Hz_inverseC(z, cdm, ol, w, e, ob):
    constC = ((cdm)/(ol+cdm) + ((ol)/(ol+cdm))*(1+z)**(3*(w+e)))**(-(e)/(w+e))
    Hz = np.sqrt( ob*(1+z)**3 + cdm*constC*(1+z)**3 +  ol*constC*(1+z)**(3*(1+w+e))) 
    return 1.0 / Hz

def dl_IDEC(cdm,ob,w,e, H0=False):
    ol = 1 - cdm - ob
    x = np.array([quad(IDE_Hz_inverseC, 0, z, args=(cdm, ol, w, e, ob))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 19) Flat w(z) with 3x parameters, \Omega_M, \omega_0 and \omega_a - DONE
def Fwz_Hz_inverse(z,om,w0,wz):
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0-wz))) * (np.exp(3*wz*z)) ) )
    return 1.0 / Hz

def dl_Fwz(om,w0,wz, H0=False):
    x = np.array([quad(Fwz_Hz_inverse, 0, z, args=(om, w0, wz))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 6) Cardassian with 3x parameters, \Omega_M, q and n
def FCa_Hz_inverse(z, om, q ,n ):
    Hz = np.sqrt( (om*((z+1)**3))*(1+(((om**(-q))-1)*((z+1)**(3*q*(n-1) ))))**(1/q)  )
    return 1.0 / Hz

def dl_FCa(om, q, n, H0=False):
    x = np.array([quad(FCa_Hz_inverse, 0, z, args=(om, q, n))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 9) Flat General Chaplygin 2x parameters, A and \alpha
def FGChap_Hz_inverse(z, A, a):
    Hz = np.sqrt((A + (1-A)*((1+z)**(3*(1+a))))**(1.0/(1+a)))
    return 1.0 / Hz

def dl_FGChap(A, a, H0=False):
    x = np.array([quad(FGChap_Hz_inverse, 0, z, args=(A, a))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 8) Chaplygin 2x parameters, A and \Omega_K
def Chap_Hz_inverse(z, A, ok):
    Hz = np.sqrt(ok*((1+z)**2)+(1-ok)*np.sqrt(A + (1-A)*((1+z)**6)))
    return 1.0 / Hz

def dl_Chap(A, ok, H0=False):
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
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist


# 10) General Chaplygin 3x parameters, \Omega_K, A and \alpha
def GChap_Hz_inverse(z, ok, A ,a):
    Hz = np.sqrt((ok*((1+z)**2)) + (1-ok)*(A + (1-A)*((1+z)**(3*(1+a))))**(1/(1+a)))
    return 1.0 / Hz

def dl_GChap(A, a, ok, H0=False):
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
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 11) DGP 2x parameters, \Omega_rc, and \Omega_K
def DGP_Hz_inverse(z, rc, ok):
    Hz = np.sqrt(ok*((1+z)**2)+ (((np.sqrt(((1 - ok - 2*(np.sqrt(rc)*np.sqrt(1-ok)))*((1+z)**3))+ rc )) + np.sqrt(rc) )**2)) 
    return 1.0 / Hz

def dl_DGP(rc, ok, H0=False):
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
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 22) New General Chaplygin 4x parameters, \Omega_K, A and \alpha, w
def NGCG_Hz_inverse(z, om, A ,a, w):
    Hz = np.sqrt(om*(1+z)**3 + ((1-om)*(1+z)**3)*(1-A*(1-(1+z)**(3*w*(1+a))))**(1/(1+a)))
    return 1.0 / Hz

def dl_NGCG(om, A, a, w, H0=False):
    x = np.array([quad(NGCG_Hz_inverse, 0, z, args=(om, A, a, w))[0] for z in zs])
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

# 11) Galileon Tracker Solution 2x parameters, \Omega_m, \Omega_g
def GAL_Hz_inverse(z, om, og):
    ok  = 1 - om - og
    Hz = np.sqrt(0.5*ok*(1+z)**2 + 0.5*om*(1+z)**3 + np.sqrt(og + 0.25*((om*(1+z)+ok)**2)*(1+z)**4))
    return 1.0 / Hz

def dl_GAL(om, og, H0=False):
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
    if H0 == True:
        h0_ = getH0(dist_mod, mu)
        return h0_
    return lum_dist

if __name__ == "__main__":
    x = 0

    #print("--- %s seconds ---" % (time.time() - start_time))