import numpy as np
from scipy.integrate import quad
import pandas as pd

# Calculates the likelihood for different models against SN+BAO/CMB constraints.
# Organised as:
## 1. SN Data to test against - This is script is just for DES5YR_UNBIN
## 2. CMB/BAO data to test against.
## 2. The likelihood functions
## 3. Models


# Current SN - data being used
DataToUse = 'DES5YR_UNBIN'
DES5YR_UNBIN = np.genfromtxt("data/%s_data.txt" % (DataToUse), names=True)
zs = DES5YR_UNBIN['zCMB']
mu = DES5YR_UNBIN['MU']
error = DES5YR_UNBIN['MUERR']
cov_arr = np.genfromtxt("data/%s_cov.txt" % (DataToUse), comments='#',dtype=None)
cov1 = cov_arr.reshape(1867,1867) 
mu_diag = np.diag(error)**2
cov = mu_diag+cov1

# Current CMB/BAO data being used - Sollerman, 2009
## Data
#f1 = 17.55 # z = 0.20
#f2 = 10.10 # z = 0.35
#f = np.array([f1, f2])
## Data_Error
#err_f1 = 0.65 # plus/minus error for f1
#err_f2 = 0.38 # plus/minus error for f2
#f_err = np.array([err_f1, err_f2])
## Redshift
#zss = np.array([0.20, 0.35])

# Current CMB/BAO data being used - Planck, 2018 & eBoss, 2020
# Data
f1 = 21.11 # z = 0.20
f2 = 5.15 # z = 0.35
f = np.array([f1, f2])
# Data_Error
err_f1 = 0.82 # plus/minus error for f1
err_f2 = 0.18 # plus/minus error for f2
f_err = np.array([err_f1, err_f2])
# Redshift
zss = np.array([0.15, 0.85])


#### The SN-Likelihood function:

def SN_cov_log_likelihood(mu_model, mu, cov):
    delta = np.array([mu_model - mu])
    inv_cov = np.linalg.inv(cov)
    deltaT = np.transpose(delta)
    chit2 = np.sum(delta @ inv_cov @ deltaT)
    B = np.sum(delta @ inv_cov)
    C = np.sum(inv_cov)
    chi2 = chit2 - (B**2 / C) + np.log(C / (2* np.pi))
    return -0.5*chi2 

#### The CMB/BAO-likelihood function:

def CMB_BAO_log_likelihood(f, f_err, model): 
    delta = model - f 
    # Might need the below stuff if we use correlation coeffecient.
    #chit2 = np.sum(delta**2 / f_err**2)
    #B = np.sum(delta/f_err**2)
    #C = np.sum(1/f_err**2)
    chi2 =  np.sum((f - model) ** 2 /f_err**2) 
    return -0.5*chi2

#### Non-Standard Models:

# 1) Flat Cosmological Constant with 1x paramater, \Omega_M 
def FLCDM_Hz_inverse(z,om, ol):
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz
    
def FLCDM(om):
    ol = 1 - om
    x = np.array([quad(FLCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zs]) # SN
    D = x
    lum_dist = D * (1 + zs) 
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov) 

    # Calculates values used for the CMB/BAO log likelihood for this model
    y = np.array([quad(FLCDM_Hz_inverse, 0, 1090, args=(om, ol))[0]]) # Last Scattering
    E = y
    ang_star = E / (1+1090)
    q = np.array([quad(FLCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zss]) # CMB/BAO
    F = q
    ang_dist = F / (1 + zss)
    Hz = np.sqrt((1 + zss) ** 2 * (om * zss + 1) - ol * zss * (zss + 2))
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO


# 2) Cosmological Constant with 2x paramaters, \Omega_M and \Omega_{\Lambda} 
def LCDM_Hz_inverse(z,om,ol):
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz

def LCDM(om, ol):
    ok = 1.0 - om - ol
    x = np.array([quad(LCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zs]) # SN 
    q = np.array([quad(LCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zss]) # CMB/BAO
    y = np.array([quad(LCDM_Hz_inverse, 0, 1090, args=(om, ol))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
    else:
        D = x
        E = y
        F = q
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$\Omega_{\Lambda}$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    Hz = np.sqrt((1 + zss) ** 2 * (om * zss + 1) - ol * zss * (zss + 2))
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO


# 3) Flat Constant wCDM with 2x paramaters, \Omega_M and \omega 
def FwCDM_Hz_inverse(z,om,w):
    ol = 1 - om
    Hz = np.sqrt((om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return 1.0 / Hz

def FwCDM(om, w):
    ol = 1 - om
    x = np.array([quad(FwCDM_Hz_inverse, 0, z, args=(om, w))[0] for z in zs]) # SN
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$\omega$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihood for this model
    y = np.array([quad(FwCDM_Hz_inverse, 0, 1090, args=(om, w))[0]]) # Last Scattering
    E = y
    ang_star = E / (1+1090)
    q = np.array([quad(FwCDM_Hz_inverse, 0, z, args=(om, w))[0] for z in zss]) #CMB/BAO
    F = q
    ang_dist = F / (1 + zss)
    Hz = np.sqrt((om*(1+zss)**(3) + ol*(1+zss)**(3*(1+w))))
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO

# 4) Constant wCDM with 3x parameters, \Omega_M, \Omega_{\Lambda} and \omega 
def wCDM_Hz_inverse(z,om,ol,w):
    omega_k = 1.0 - om - ol
    Hz = np.sqrt((omega_k*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return 1.0 / (Hz)
    
def wCDM(om,ol,w):
    ok = 1.0 - om - ol
    x = np.array([quad(wCDM_Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zs]) # SN
    q = np.array([quad(wCDM_Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zss]) # CMB/BAO
    y = np.array([quad(wCDM_Hz_inverse, 0, 1090, args=(om, ol, w))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
    else:
        D = x
        E = y
        F = q
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$\Omega_{\Lambda}$",r"$\omega$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    omega_k = 1.0 - om - ol
    Hz = np.sqrt((omega_k*(1+zss)**(2) + om*(1+zss)**(3) + ol*(1+zss)**(3*(1+w))))
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO

# 5) Flat w(a) with 3x parameters, \Omega_M, \omega_0 and \omega_a 
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
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    y = np.array([quad(Fwa_Hz_inverse, 0, 1090, args=(om, w0, wa))[0]]) # Last Scattering
    E = y
    q = np.array([quad(Fwa_Hz_inverse, 0, z, args=(om, w0, wa))[0] for z in zss]) #CMB/BAO
    F = q
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+zss)**(3)) + (ol * ((1+zss)**(3*(1+w0+wa))) * (np.exp(-3*wa*(1-((1+zss)**(-1))))) ) )
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO

# 6) IDE1 Q = H e rho_x
def IDE_Hz_inverse1(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(cdm*(1+z)**3 + ol*(1-(e/(3*w + e)))*(1+z)**((3*(1+w)+e)) + ok*(1+z)**2) 
    return 1.0 / Hz

def IDE1(cdm,ol,w,e):
    ok = 1 -ol - cdm
    x = np.array([quad(IDE_Hz_inverse1, 0, z, args=(cdm, ol, w, e))[0] for z in zs]) # SN
    q = np.array([quad(IDE_Hz_inverse1, 0, z, args=(cdm, ol, w, e))[0] for z in zss]) # CMB/BAO
    y = np.array([quad(IDE_Hz_inverse1, 0, 1090, args=(cdm, ol, w, e))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
    else:
        D = x
        E = y
        F = q
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = [r"$\Omega_{CDM}$", r"$\Omega_{DE}$", r"$\omega$", r"$\epsilon$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    Hz = np.sqrt(cdm*(1+zss)**3 + ol*(1-(e/(3*w + e)))*(1+zss)**((3*(1+w)+e)) + ok*(1+zss)**2) 
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO

# 7) IDE2 Q = H e rho_c
def IDE_Hz_inverse2(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(cdm*((1+z)**(3-e)) * (1-(e)/(3*w +e )) + ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
    return 1.0 / Hz

def IDE2(cdm,ol,w,e):
    ok = 1 -ol - cdm
    x = np.array([quad(IDE_Hz_inverse2, 0, z, args=(cdm, ol, w, e))[0] for z in zs]) # SN
    q = np.array([quad(IDE_Hz_inverse2, 0, z, args=(cdm, ol, w, e))[0] for z in zss]) # CMB/BAO
    y = np.array([quad(IDE_Hz_inverse2, 0, 1090, args=(cdm, ol, w, e))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
    else:
        D = x
        E = y
        F = q
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = [r"$\Omega_{CDM}$", r"$\Omega_{DE}$", r"$\omega$", r"$\epsilon$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    Hz = np.sqrt(cdm*((1+zss)**(3-e)) * (1-(e)/(3*w +e )) + ol*(1+zss)**(3*(1+w)) + ok*(1+zss)**2) 
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO

# 8) IDE4 Q = H e [rho_c * rho_x / (rho_c + rho_x)]
def IDE_Hz_inverse4(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    IDE4_const = ( ((1+z)**(-1*(e+(3*w)))) + (ol/cdm) )**(-e/(e+(3*w)))
    Hz = np.sqrt( (cdm*IDE4_const*(1+z)**(3-e)) + IDE4_const*ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
    return 1.0 / Hz

def IDE4(cdm,ol,w,e):
    ok = 1 -ol - cdm
    x = np.array([quad(IDE_Hz_inverse4, 0, z, args=(cdm, ol, w, e))[0] for z in zs]) # SN
    q = np.array([quad(IDE_Hz_inverse4, 0, z, args=(cdm, ol, w, e))[0] for z in zss]) # CMB/BAO
    y = np.array([quad(IDE_Hz_inverse4, 0, 1090, args=(cdm, ol, w, e))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
    else:
        D = x
        E = y
        F = q
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = [r"$\Omega_{CDM}$", r"$\Omega_{DE}$", r"$\omega$", r"$\epsilon$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    IDE4_const = ( ((1+zss)**(-1*(e+(3*w)))) + (ol/cdm) )**(-e/(e+(3*w)))
    Hz = np.sqrt( (cdm*IDE4_const*(1+zss)**(3-e)) + IDE4_const*ol*(1+zss)**(3*(1+w)) + ok*(1+zss)**2) 
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO

# 9) Flat w(z) with 3x parameters, \Omega_M, \omega_0 and \omega_a 
def Fwz_Hz_inverse(z,om,w0,wz):
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0-wz))) * (np.exp(3*wz*z)) ) )
    return 1.0 / Hz

def Fwz(om,w0,wz):
    x = np.array([quad(Fwa_Hz_inverse, 0, z, args=(om, w0, wz))[0] for z in zs]) # SN
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$w_0$","$w_z$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    y = np.array([quad(Fwz_Hz_inverse, 0, 1090, args=(om, w0, wz))[0]]) # Last Scattering
    E = y
    q = np.array([quad(Fwz_Hz_inverse, 0, z, args=(om, w0, wz))[0] for z in zss]) # CMB/BAO
    F = q
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+zss)**(3)) + (ol * ((1+zss)**(3*(1+w0-wz))) * (np.exp(3*wz*zss)) ) )
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO

# 10) Cardassian with 3x parameters, \Omega_M, q and n
def FCa_Hz_inverse(z, om, q ,n ):
    Hz = np.sqrt(
        (om*((z+1)**3))*(1+(((om**(-q))-1)*((z+1)**(3*q*(n-1)))))**(1/q))
    return 1.0 / Hz

def FCa(om, q, n):
    x = np.array([quad(FCa_Hz_inverse, 0, z, args=(om, q, n))[0] for z in zs]) # SN
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_m$","$q$","$n$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    y = np.array([quad(FCa_Hz_inverse, 0, 1090, args=(om, q, n))[0]]) # Last Scattering
    E = y
    q = np.array([quad(FCa_Hz_inverse, 0, z, args=(om, q, n))[0] for z in zss]) # CMB/BAO
    F = q
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    Hz = np.sqrt((om*((zss+1)**3))*(1+(((om**(-q))-1)*((zss+1)**(3*q*(n-1)))))**(1/q))
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO

# 11) Flat General Chaplygin 2x parameters, A and \alpha
def FGChap_Hz_inverse(z, A, a):
    Hz = np.sqrt((A + (1-A)*((1+z)**(3*(1+a))))**(1.0/(1+a)))
    return 1.0 / Hz

def FGChap(A, a):
    x = np.array([quad(FGChap_Hz_inverse, 0, z, args=(A, a))[0] for z in zs]) # SN
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$A$", r"$\alpha$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    y = np.array([quad(FGChap_Hz_inverse, 0, 1090, args=(A, a))[0]]) # Last Scattering
    E = y
    q = np.array([quad(FGChap_Hz_inverse, 0, z, args=(A, a))[0] for z in zss]) # CMB/BAO
    F = q
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    Hz = np.sqrt((A + (1-A)*((1+zss)**(3*(1+a))))**(1.0/(1+a)))
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO

# 12) Chaplygin 2x parameters, A and \Omega_K
def Chap_Hz_inverse(z, A, ok):
    Hz = np.sqrt(ok*((1+z)**2)+(1-ok)*np.sqrt(A + (1-A)*((1+z)**6)))
    return 1.0 / Hz

def Chap(A, ok):
    x = np.array([quad(Chap_Hz_inverse, 0, z, args=(A, ok))[0] for z in zs]) # SN
    q = np.array([quad(Chap_Hz_inverse, 0, z, args=(A, ok))[0] for z in zss]) # CMB/BAO
    y = np.array([quad(Chap_Hz_inverse, 0, 1090, args=(A, ok))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
    else:
        D = x
        E = y
        F = q
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$A$","$\Omega_k$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    Hz = np.sqrt(ok*((1+zss)**2)+(1-ok)*np.sqrt(A + (1-A)*((1+zss)**6)))
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO


# 13) General Chaplygin 3x parameters, \Omega_K, A and \alpha
def GChap_Hz_inverse(z, ok, A ,a):
    Hz = np.sqrt((ok*((1+z)**2)) + (1-ok)*(A + (1-A)*((1+z)**(3*(1+a))))**(1/(1+a)))
    return 1.0 / Hz

def GChap(ok, A, a):
    x = np.array([quad(GChap_Hz_inverse, 0, z, args=(ok, A, a))[0] for z in zs]) # SN
    q = np.array([quad(GChap_Hz_inverse, 0, z, args=(ok, A, a))[0] for z in zss]) # CMB/BAO
    y = np.array([quad(GChap_Hz_inverse, 0, 1090, args=(ok, A, a))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
    else:
        D = x
        E = y
        F = q
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["$\Omega_K$","$A$",r"$\alpha$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    Hz = np.sqrt((ok*((1+zss)**2)) + (1-ok)*(A + (1-A)*((1+zss)**(3*(1+a))))**(1/(1+a)))
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO

# 14) DGP 2x parameters, \Omega_rc, and \Omega_K
def DGP_Hz_inverse(z, rc, ok):
    Hz = np.sqrt(ok*((1+z)**2)+ (((np.sqrt(((1 - ok - 2*(np.sqrt(rc)*np.sqrt(1-ok)))*((1+z)**3))+ rc )) + np.sqrt(rc) )**2)) 
    return 1.0 / Hz

def DGP(rc, ok):
    x = np.array([quad(DGP_Hz_inverse, 0, z, args=(rc, ok))[0] for z in zs]) # SN
    q = np.array([quad(DGP_Hz_inverse, 0, z, args=(rc, ok))[0] for z in zss]) # CMB/BAO
    y = np.array([quad(DGP_Hz_inverse, 0, 1090, args=(rc, ok))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        D = R0 * np.sin(x / R0)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        D = R0 * np.sinh(x / R0)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
    else:
        D = x
        E = y
        F = q
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = [r"$\Omega_{rc}$", r"$\Omega_K$"]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    Hz = np.sqrt(ok*((1+zss)**2)+ (((np.sqrt(((1 - ok - 2*(np.sqrt(rc)*np.sqrt(1-ok)))*((1+zss)**3))+ rc )) + np.sqrt(rc) )**2)) 
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO

# 15) New General Chaplygin 4x parameters, om, A and \alpha, w - this is the flat version
def NGCG_Hz_inverse(z, om, A ,a, w):
    Hz = np.sqrt(om*(1+z)**3 + ((1-om)*(1+z)**3)*(1-A*(1-(1+z)**(3*w*(1+a))))**(1/(1+a)))
    return 1.0 / Hz

def NGCG(om, A, a, w):
    x = np.array([quad(NGCG_Hz_inverse, 0, z, args=(om, A, a, w))[0] for z in zs]) # SN
    D = x
    lum_dist = D * (1 + zs)
    dist_mod = 5 * np.log10(lum_dist)
    label = ["\Omega_m", "$A$",r"$\alpha$","$\Omega_K$",]
    log_SN = SN_cov_log_likelihood(dist_mod, mu, cov)

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    y = np.array([quad(NGCG_Hz_inverse, 0, 1090, args=(om, A, a,w))[0]]) # Last Scattering
    E = y
    q = np.array([quad(NGCG_Hz_inverse, 0, z, args=(om, A, a, w))[0] for z in zss]) # CMB/BAO
    F = q
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zss)
    Hz = np.sqrt(om*(1+zss)**3 + ((1-om)*(1+zss)**3)*(1-A*(1-(1+zss)**(3*w*(1+a))))**(1/(1+a)))
    D_V = ((1 + zss)**2 * ang_dist**2 * (zss)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_CMB_BAO = CMB_BAO_log_likelihood(f, f_err, model)
    return log_SN + log_CMB_BAO


if __name__ == "__main__":
    #logp = LCDM(0.31,0.7)
    logp = FwCDM(0.3,-1)
    print(logp)