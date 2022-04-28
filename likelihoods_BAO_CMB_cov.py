import numpy as np
from scipy.integrate import quad
import pandas as pd

# Calculates the likelihood for different models against BAO/CMB data
# Organised as:
## 1. CMB/BAO Data to test against
## 2. The likelihood functions
## 3. Models

# Current CMB/BAO data being used - Planck, 2018 & eBoss, 2020

# f data f = DM/DV - uncorrelated
f = np.array([21.12893607291012, 5.15136651780974]) # Data
f_err = np.array([0.7979734940061242, 0.2503450640287214])
zs = np.array([0.15, 0.845]) # Redshift

# For the following data points, some have correlations some do not.
# The redshifts that we cant to calculate the likelihood for though are:
zm = np.array([0.38, 0.51, 0.698, 1.48, 2.334, 2.334])
zh = np.array([0.38, 0.51, 0.698, 1.48, 2.334, 2.334])

### Uncorrelated DM/DM data:
g = np.array([2.519846652953153, 2.51995219626846])
g_err = np.array([0.12897345549113046, 0.10960173956873172])
### Uncorrelated DM/DH data:
h = np.array([10.581288504779634, 10.417287376153663])
h_err = np.array([0.33951431286569206, 0.3630132470749218])

### Correlated data for BOSS:
BOSS_data = np.array([9.219682983938531, 3.7771256247295284, 7.05934025180447, 4.228016721152631])
BOSS_cov = np.array([[ 0.02396093,-0.00642275,0.0076587,-0.00202567],[-0.00642275,0.01225848,-0.00170161,0.00520206],[0.0076587 ,-0.00170161  ,0.01200653 ,-0.00461535],[-0.00202567 , 0.00520206 ,-0.00461535 , 0.01188892]])
### Correlated data for LRG:
LRG_data = np.array([5.283544465695126, 4.882334223748529])
LRG_cov = np.array([[ 0.00966889, -0.00413278],[-0.00413278 , 0.01832329]])
### Correlated data for QU:
QU_data = np.array([3.0746877839455014, 7.115262828209696,])
QU_cov = np.array([[0.00648069, 0.00936799],[0.00936799, 0.08816128]])

#### The CMB/BAO-likelihood function:

# Using for the uncorrelated data
def CMB_BAO_log_likelihood(f, f_err, model): 
    delta = model - f
    chit2 = np.sum(delta**2 / f_err**2)
    B = np.sum(delta/f_err**2)
    C = np.sum(1/f_err**2)
    chi2 = chit2 #- (B**2 / C) + np.log(C/(2* np.pi))
    return -0.5*chi2

# For the correlated data
def CMB_BAO_cov_log_likelihood(mu_model, mu, cov):
    delta = mu_model - mu
    inv_cov = np.linalg.inv(cov)
    deltaT = np.transpose(delta)
    chit2 = np.sum(delta @ inv_cov @ deltaT)
    B = np.sum(delta @ inv_cov)
    C = np.sum(inv_cov)
    chi2 = chit2 #- (B**2 / C) + np.log(C / (2* np.pi))
    return -0.5*chi2 

#### Non-Standard Models:

# 1) Flat Cosmological Constant with 1x paramater, \Omega_M 
def FLCDM_Hz_inverse(z,om, ol):
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz
    
def FLCDM(om):
    ol = 1 - om
    # Calculates values used for the CMB/BAO log likelihood for this model
    y = np.array([quad(FLCDM_Hz_inverse, 0, 1090, args=(om, ol))[0]]) # Last Scattering
    E = y
    ang_star = E / (1+1090)

    # calculates the values used for Dv/rd data - zs array
    q = np.array([quad(FLCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zs])
    F = q
    ang_dist = F / (1 + zs)
    Hz = np.sqrt((1 + zs) ** 2 * (om * zs + 1) - ol * zs * (zs + 2))
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model) # likelihood for dv/rd data

    # calculates the values used for Dm/rd data - zm array
    m = np.array([quad(FLCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zm])
    M = m
    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/( np.sqrt((1 + zh) ** 2 * (om * zh + 1) - ol * zh * (zh+ 2)))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO


# 2) Cosmological Constant with 2x paramaters, \Omega_M and \Omega_{\Lambda} 
def LCDM_Hz_inverse(z,om,ol):
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return 1.0 / Hz

def LCDM(om, ol):
    ok = 1.0 - om - ol
    q = np.array([quad(LCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zs]) #dv/rd data
    y = np.array([quad(LCDM_Hz_inverse, 0, 1090, args=(om, ol))[0]]) # last scattering
    m = np.array([quad(LCDM_Hz_inverse, 0, z, args=(om, ol))[0] for z in zm]) # dm/rd data
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
        M = R0 * np.sin(m / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
        M = R0 * np.sinh(m / R0)
    else:
        E = y
        F = q
        M = m
    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt((1 + zs) ** 2 * (om * zs + 1) - ol * zs * (zs + 2))
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    # calculates the values used for Dm/rd data - zm array
    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/( np.sqrt((1 + zh) ** 2 * (om * zh + 1) - ol * zh * (zh + 2)))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO =  log_BOSS + log_dh + log_dm + log_dv + log_QU  + log_LRG 
    return log_CMB_BAO

# 3) Flat Constant wCDM with 2x paramaters, \Omega_M and \omega 
def FwCDM_Hz_inverse(z,om,w):
    ol = 1 - om
    Hz = np.sqrt((om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return 1.0 / Hz

def FwCDM(om, w):
    ol = 1 - om
    # Calculates values used for the CMB/BAO log likelihood for this model
    y = np.array([quad(FwCDM_Hz_inverse, 0, 1090, args=(om, w))[0]]) # Last Scattering
    E = y
    ang_star = E / (1+1090)
    q = np.array([quad(FwCDM_Hz_inverse, 0, z, args=(om, w))[0] for z in zs]) 
    F = q
    ang_dist = F / (1 + zs)
    Hz = np.sqrt((om*(1+zs)**(3) + ol*(1+zs)**(3*(1+w))))
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    # calculates the values used for Dm/rd data - zm array
    m = np.array([quad(FwCDM_Hz_inverse, 0, z, args=(om, w))[0] for z in zm]) 
    M = m
    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt((om*(1+zh)**(3) + ol*(1+zh)**(3*(1+w)))))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 4) Constant wCDM with 3x parameters, \Omega_M, \Omega_{\Lambda} and \omega 
def wCDM_Hz_inverse(z,om,ol,w):
    omega_k = 1.0 - om - ol
    Hz = np.sqrt((omega_k*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return 1.0 / (Hz)
    
def wCDM(om,ol,w):
    ok = 1.0 - om - ol
    q = np.array([quad(wCDM_Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zs]) # dv/rd
    m = np.array([quad(wCDM_Hz_inverse, 0, z, args=(om, ol, w))[0] for z in zm]) # dm/rd 
    y = np.array([quad(wCDM_Hz_inverse, 0, 1090, args=(om, ol, w))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
        M = R0 * np.sin(m / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
        M = R0 * np.sinh(m / R0)
    else:
        E = y
        F = q
        M = m

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    omega_k = 1.0 - om - ol
    Hz = np.sqrt((omega_k*(1+zs)**(2) + om*(1+zs)**(3) + ol*(1+zs)**(3*(1+w))))
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt((omega_k*(1+zh)**(2) + om*(1+zh)**(3) + ol*(1+zh)**(3*(1+w)))))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 5) Flat w(a) with 3x parameters, \Omega_M, \omega_0 and \omega_a 
def Fwa_Hz_inverse(z,om,w0,wa):
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0+wa))) * (np.exp(-3*wa*(1-((1+z)**(-1))))) ) )
    return 1.0 / Hz

def Fwa(om, w0, wa):
    # Calculates values used for the CMB/BAO log likelihoodfor this model
    y = np.array([quad(Fwa_Hz_inverse, 0, 1090, args=(om, w0, wa))[0]]) # Last Scattering
    E = y
    q = np.array([quad(Fwa_Hz_inverse, 0, z, args=(om, w0, wa))[0] for z in zs]) # dv/rd data
    F = q
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+zs)**(3)) + (ol * ((1+zs)**(3*(1+w0+wa))) * (np.exp(-3*wa*(1-((1+zs)**(-1))))) ) )
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    m = np.array([quad(Fwa_Hz_inverse, 0, z, args=(om, w0, wa))[0] for z in zm]) # dm/rd data
    M = m
    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt( (om*(1+zh)**(3)) + (ol * ((1+zh)**(3*(1+w0+wa))) * (np.exp(-3*wa*(1-((1+zh)**(-1))))) ) ))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 6) IDE1 Q = H e rho_x
def IDE_Hz_inverse1(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(cdm*(1+z)**3 + ol*(1-(e/(3*w + e)))*(1+z)**((3*(1+w)+e)) + ok*(1+z)**2) 
    return 1.0 / Hz

def IDE1(cdm,ol,w,e):
    ok = 1 -ol - cdm
    q = np.array([quad(IDE_Hz_inverse1, 0, z, args=(cdm, ol, w, e))[0] for z in zs]) # dv/rd data
    m = np.array([quad(IDE_Hz_inverse1, 0, z, args=(cdm, ol, w, e))[0] for z in zm]) # dm/rd data
    y = np.array([quad(IDE_Hz_inverse1, 0, 1090, args=(cdm, ol, w, e))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
        M = R0 * np.sin(m / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
        M = R0 * np.sinh(m / R0)
    else:
        E = y
        F = q
        M = m

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt(cdm*(1+zs)**3 + ol*(1-(e/(3*w + e)))*(1+zs)**((3*(1+w)+e)) + ok*(1+zs)**2) 
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt(cdm*(1+zh)**3 + ol*(1-(e/(3*w + e)))*(1+zh)**((3*(1+w)+e)) + ok*(1+zh)**2) )
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 7) IDE2 Q = H e rho_c
def IDE_Hz_inverse2(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(cdm*((1+z)**(3-e)) * (1-(e)/(3*w +e )) + ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
    return 1.0 / Hz

def IDE2(cdm,ol,w,e):
    ok = 1 -ol - cdm
    q = np.array([quad(IDE_Hz_inverse2, 0, z, args=(cdm, ol, w, e))[0] for z in zs]) 
    m = np.array([quad(IDE_Hz_inverse2, 0, z, args=(cdm, ol, w, e))[0] for z in zm]) 
    y = np.array([quad(IDE_Hz_inverse2, 0, 1090, args=(cdm, ol, w, e))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
        M = R0 * np.sin(m / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
        M = R0 * np.sinh(m / R0)
    else:
        E = y
        F = q
        M = m
    
    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt(cdm*((1+zs)**(3-e)) * (1-(e)/(3*w +e )) + ol*(1+zs)**(3*(1+w)) + ok*(1+zs)**2) 
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt(cdm*((1+zh)**(3-e)) * (1-(e)/(3*w +e )) + ol*(1+zh)**(3*(1+w)) + ok*(1+zh)**2))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 8) IDE4 Q = H e [rho_c * rho_x / (rho_c + rho_x)]
def IDE_Hz_inverse4(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    IDE4_const = ( ((1+z)**(-1*(e+(3*w)))) + (ol/cdm) )**(-e/(e+(3*w)))
    Hz = np.sqrt( (cdm*IDE4_const*(1+z)**(3-e)) + IDE4_const*ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
    return 1.0 / Hz

def IDE4(cdm,ol,w,e):
    ok = 1 -ol - cdm
    q = np.array([quad(IDE_Hz_inverse4, 0, z, args=(cdm, ol, w, e))[0] for z in zs])
    m = np.array([quad(IDE_Hz_inverse4, 0, z, args=(cdm, ol, w, e))[0] for z in zm])
    y = np.array([quad(IDE_Hz_inverse4, 0, 1090, args=(cdm, ol, w, e))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
        M = R0 * np.sin(m / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
        M = R0 * np.sinh(m / R0)
    else:
        E = y
        F = q
        M = m

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    IDE4_const = ( ((1+zs)**(-1*(e+(3*w)))) + (ol/cdm) )**(-e/(e+(3*w)))
    Hz = np.sqrt( (cdm*IDE4_const*(1+zs)**(3-e)) + IDE4_const*ol*(1+zs)**(3*(1+w)) + ok*(1+zs)**2) 
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    IDE4_const1 = ( ((1+zh)**(-1*(e+(3*w)))) + (ol/cdm) )**(-e/(e+(3*w)))
    Dh = 1/(np.sqrt( (cdm*IDE4_const1*(1+zh)**(3-e)) + IDE4_const1*ol*(1+zh)**(3*(1+w)) + ok*(1+zh)**2) )
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 9) Flat w(z) with 3x parameters, \Omega_M, \omega_0 and \omega_a 
def Fwz_Hz_inverse(z,om,w0,wz):
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0-wz))) * (np.exp(3*wz*z)) ) )
    return 1.0 / Hz

def Fwz(om,w0,wz):
    # Calculates values used for the CMB/BAO log likelihoodfor this model
    y = np.array([quad(Fwz_Hz_inverse, 0, 1090, args=(om, w0, wz))[0]]) # Last Scattering
    E = y
    q = np.array([quad(Fwz_Hz_inverse, 0, z, args=(om, w0, wz))[0] for z in zs]) # dv/rd data
    F = q
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+zs)**(3)) + (ol * ((1+zs)**(3*(1+w0-wz))) * (np.exp(3*wz*zs)) ) )
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    m = np.array([quad(Fwz_Hz_inverse, 0, z, args=(om, w0, wz))[0] for z in zm]) # dm/rd data
    M = m
    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt( (om*(1+zh)**(3)) + (ol * ((1+zh)**(3*(1+w0-wz))) * (np.exp(3*wz*zh)) ) ))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 10) Cardassian with 3x parameters, \Omega_M, q and n
def FCa_Hz_inverse(z, om, q ,n ):
    Hz = np.sqrt(
        (om*((z+1)**3))*(1+(((om**(-q))-1)*((z+1)**(3*q*(n-1)))))**(1/q))
    return 1.0 / Hz

def FCa(om, q, n):
    # Calculates values used for the CMB/BAO log likelihoodfor this model
    y = np.array([quad(FCa_Hz_inverse, 0, 1090, args=(om, q, n))[0]]) # Last Scattering
    E = y
    v = np.array([quad(FCa_Hz_inverse, 0, z, args=(om, q, n))[0] for z in zs]) #dv/rd data
    F = v
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt((om*((zs+1)**3))*(1+(((om**(-q))-1)*((zs+1)**(3*q*(n-1)))))**(1/q))
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    m = np.array([quad(FCa_Hz_inverse, 0, z, args=(om, q, n))[0] for z in zm]) #dm/rd data
    M = m
    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt((om*((zh+1)**3))*(1+(((om**(-q))-1)*((zh+1)**(3*q*(n-1)))))**(1/q)))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 11) Flat General Chaplygin 2x parameters, A and \alpha
def FGChap_Hz_inverse(z, A, a):
    Hz = np.sqrt((A + (1-A)*((1+z)**(3*(1+a))))**(1.0/(1+a)))
    return 1.0 / Hz

def FGChap(A, a):
    # Calculates values used for the CMB/BAO log likelihoodfor this model
    y = np.array([quad(FGChap_Hz_inverse, 0, 1090, args=(A, a))[0]]) # Last Scattering
    E = y
    q = np.array([quad(FGChap_Hz_inverse, 0, z, args=(A, a))[0] for z in zs]) 
    F = q
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt((A + (1-A)*((1+zs)**(3*(1+a))))**(1.0/(1+a)))
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    m = np.array([quad(FGChap_Hz_inverse, 0, z, args=(A, a))[0] for z in zm]) #dm/rd data
    M = m
    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt((A + (1-A)*((1+zh)**(3*(1+a))))**(1.0/(1+a))))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 12) Chaplygin 2x parameters, A and \Omega_K
def Chap_Hz_inverse(z, A, ok):
    Hz = np.sqrt(ok*((1+z)**2)+(1-ok)*np.sqrt(A + (1-A)*((1+z)**6)))
    return 1.0 / Hz

def Chap(A, ok):
    q = np.array([quad(Chap_Hz_inverse, 0, z, args=(A, ok))[0] for z in zs]) #dv/rd data
    m = np.array([quad(Chap_Hz_inverse, 0, z, args=(A, ok))[0] for z in zm])  #dm/rd data
    y = np.array([quad(Chap_Hz_inverse, 0, 1090, args=(A, ok))[0]]) # last scattering
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
        M = R0 * np.sin(m / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
        M = R0 * np.sinh(m / R0)
    else:
        E = y
        F = q
        M = m


    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt(ok*((1+zs)**2)+(1-ok)*np.sqrt(A + (1-A)*((1+zs)**6)))
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt(ok*((1+zh)**2)+(1-ok)*np.sqrt(A + (1-A)*((1+zh)**6))))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 13) General Chaplygin 3x parameters, \Omega_K, A and \alpha
def GChap_Hz_inverse(z, ok, A ,a):
    Hz = np.sqrt((ok*((1+z)**2)) + (1-ok)*(A + (1-A)*((1+z)**(3*(1+a))))**(1/(1+a)))
    return 1.0 / Hz

def GChap(A, a, ok):
    q = np.array([quad(GChap_Hz_inverse, 0, z, args=(ok, A, a))[0] for z in zs]) #dv/rd data
    m = np.array([quad(GChap_Hz_inverse, 0, z, args=(ok, A, a))[0] for z in zm]) #dm/rd data 
    y = np.array([quad(GChap_Hz_inverse, 0, 1090, args=(ok, A, a))[0]]) 
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
        M = R0 * np.sin(m / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
        M = R0 * np.sinh(m / R0)
    else:
        E = y
        F = q
        M = m

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt((ok*((1+zs)**2)) + (1-ok)*(A + (1-A)*((1+zs)**(3*(1+a))))**(1/(1+a)))
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt((ok*((1+zh)**2)) + (1-ok)*(A + (1-A)*((1+zh)**(3*(1+a))))**(1/(1+a))))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 14) DGP 2x parameters, \Omega_rc, and \Omega_K
def DGP_Hz_inverse(z, rc, ok):
    Hz = np.sqrt(ok*((1+z)**2)+ (((np.sqrt(((1 - ok - 2*(np.sqrt(rc)*np.sqrt(1-ok)))*((1+z)**3))+ rc )) + np.sqrt(rc) )**2)) 
    return 1.0 / Hz

def DGP(rc, ok):
    q = np.array([quad(DGP_Hz_inverse, 0, z, args=(rc, ok))[0] for z in zs]) #dv/rd data
    m = np.array([quad(DGP_Hz_inverse, 0, z, args=(rc, ok))[0] for z in zm]) #dm/rd data
    y = np.array([quad(DGP_Hz_inverse, 0, 1090, args=(rc, ok))[0]]) 
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
        M = R0 * np.sin(m / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
        M = R0 * np.sinh(m / R0)
    else:
        E = y
        F = q
        M = m

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt(ok*((1+zs)**2)+ (((np.sqrt(((1 - ok - 2*(np.sqrt(rc)*np.sqrt(1-ok)))*((1+zs)**3))+ rc )) + np.sqrt(rc) )**2)) 
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt(ok*((1+zh)**2)+ (((np.sqrt(((1 - ok - 2*(np.sqrt(rc)*np.sqrt(1-ok)))*((1+zh)**3))+ rc )) + np.sqrt(rc) )**2)) )
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 15) New General Chaplygin 4x parameters, om, A and \alpha, w - this is the flat version
def NGCG_Hz_inverse(z, om, A ,a, w):
    Hz = np.sqrt(om*(1+z)**3 + ((1-om)*(1+z)**3)*(1-A*(1-(1+z)**(3*w*(1+a))))**(1/(1+a)))
    return 1.0 / Hz

def NGCG(om, A, a, w):
    x = np.array([quad(NGCG_Hz_inverse, 0, z, args=(om, A, a, w))[0] for z in zs]) # dv/rd data
    D = x

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    y = np.array([quad(NGCG_Hz_inverse, 0, 1090, args=(om, A, a,w))[0]]) # Last Scattering
    E = y
    ang_star = E / (1+1090)
    ang_dist = D / (1 + zs)
    Hz = np.sqrt(om*(1+zs)**3 + ((1-om)*(1+zs)**3)*(1-A*(1-(1+zs)**(3*w*(1+a))))**(1/(1+a)))
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    m = np.array([quad(NGCG_Hz_inverse, 0, z, args=(om, A, a, w))[0] for z in zm]) #dm/rd data
    M = m
    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt(om*(1+zh)**3 + ((1-om)*(1+zh)**3)*(1-A*(1-(1+zh)**(3*w*(1+a))))**(1/(1+a))))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO


# 11) Galileon Tracker Solution 2x parameters, \Omega_m, \Omega_g
def GAL_Hz_inverse(z, om, og):
    ok  = 1 - om - og
    Hz = np.sqrt(0.5*ok*(1+z)**2 + 0.5*om*(1+z)**3 + np.sqrt(og + 0.25*((om*(1+z)+ok)**2)*(1+z)**4))
    return 1.0 / Hz

def GAL(om, og):
    ok = 1-om-og
    q = np.array([quad(GAL_Hz_inverse, 0, z, args=(om, og))[0] for z in zs]) # dv/rd data
    m = np.array([quad(GAL_Hz_inverse, 0, z, args=(om, og))[0] for z in zm]) # dm/rd data
    y = np.array([quad(GAL_Hz_inverse, 0, 1090, args=(om, og))[0]]) 
    if ok < 0.0:
        R0 = 1 / np.sqrt(-ok)
        E = R0 * np.sin(y / R0)
        F = R0 * np.sin(q / R0)
        M = R0 * np.sin(m / R0)
    elif ok > 0.0:
        R0 = 1 / np.sqrt(ok)
        E = R0 * np.sinh(y / R0)
        F = R0 * np.sinh(q / R0)
        M = R0 * np.sinh(m / R0)
    else:
        E = y
        F = q
        M = m

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt(0.5*ok*(1+zs)**2 + 0.5*om*(1+zs)**3 + np.sqrt(og + 0.25*((om*(1+zs)+ok)**2)*(1+zs)**4))
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt(0.5*ok*(1+zh)**2 + 0.5*om*(1+zh)**3 + np.sqrt(og + 0.25*((om*(1+zh)+ok)**2)*(1+zh)**4)))
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 6) IDE1 Q = H e rho_x
def IDEA_Hz_inverse(z, cdm, ol, w, e):
    Hz = np.sqrt(cdm*(1+z)**3 + ol*( ((e)/(w+e))*(1+z)**3 + ((w)/(w+e))*(1+z)**(3*(1+w+e))  )) 
    return 1.0 / Hz

def IDEA(cdm,w,e):
    ol = 1 - cdm
    q = np.array([quad(IDEA_Hz_inverse, 0, z, args=(cdm, ol, w, e))[0] for z in zs]) # dv/rd data
    m = np.array([quad(IDEA_Hz_inverse, 0, z, args=(cdm, ol, w, e))[0] for z in zm]) # dm/rd data
    y = np.array([quad(IDEA_Hz_inverse, 0, 1090, args=(cdm, ol, w, e))[0]]) # last scattering
    E = y
    F = q
    M = m

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt(cdm*(1+zs)**3 + ol*( ((e)/(w+e))*(1+zs)**3 + ((w)/(w+e))*(1+zs)**(3*(1+w+e))  )) 
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt(cdm*(1+zh)**3 + ol*( ((e)/(w+e))*(1+zh)**3 + ((w)/(w+e))*(1+zh)**(3*(1+w+e))  )) )
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 7) IDE2 Q = H e rho_c
def IDEB_Hz_inverse(z, cdm, ol, w, e, ob):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(ob*(1+z)**(3)+ ol*(1+z)**(3*(1+w)) + cdm*(((e)/(w+e))*(1+z)**(3*(1+w))  + ((w)/(w+e))*(1+z)**(3*(1-e)))) 
    return 1.0 / Hz

def IDEB(cdm,ob,w,e):
    ol = 1 -ob - cdm
    q = np.array([quad(IDEB_Hz_inverse, 0, z, args=(cdm, ol, w, e, ob))[0] for z in zs]) 
    m = np.array([quad(IDEB_Hz_inverse, 0, z, args=(cdm, ol, w, e, ob))[0] for z in zm]) 
    y = np.array([quad(IDEB_Hz_inverse, 0, 1090, args=(cdm, ol, w, e, ob))[0]]) # last scattering

    E = y
    F = q
    M = m
    
    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt(ob*(1+zs)**(3)+ ol*(1+zs)**(3*(1+w)) + cdm*(((e)/(w+e))*(1+zs)**(3*(1+w))  + ((w)/(w+e))*(1+zs)**(3*(1-e)))) 
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt(ob*(1+zh)**(3)+ ol*(1+zh)**(3*(1+w)) + cdm*(((e)/(w+e))*(1+zh)**(3*(1+w))  + ((w)/(w+e))*(1+zh)**(3*(1-e)))) )
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO


# 7) IDE2 Q = H e rho_c
def IDEB_Hz_inverse_2(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(ol*(1+z)**(3*(1+w)) + cdm*(((e)/(w+e))*(1+z)**(3*(1+w))  + ((w)/(w+e))*(1+z)**(3*(1-e)))) 
    return 1.0 / Hz

def IDEB_2(cdm,w,e):
    ol = 1 - cdm
    q = np.array([quad(IDEB_Hz_inverse_2, 0, z, args=(cdm, ol, w, e))[0] for z in zs]) 
    m = np.array([quad(IDEB_Hz_inverse_2, 0, z, args=(cdm, ol, w, e))[0] for z in zm]) 
    y = np.array([quad(IDEB_Hz_inverse_2, 0, 1090, args=(cdm, ol, w, e))[0]]) # last scattering

    E = y
    F = q
    M = m
    
    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    Hz = np.sqrt(ol*(1+zs)**(3*(1+w)) + cdm*(((e)/(w+e))*(1+zs)**(3*(1+w))  + ((w)/(w+e))*(1+zs)**(3*(1-e)))) 
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    Dh = 1/(np.sqrt(ol*(1+zh)**(3*(1+w)) + cdm*(((e)/(w+e))*(1+zh)**(3*(1+w))  + ((w)/(w+e))*(1+zh)**(3*(1-e)))) )
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO

# 8) IDE4 Q = H e [rho_c * rho_x / (rho_c + rho_x)]
def IDEC_Hz_inverse(z, cdm, ol, w, e, ob):
    constC = ((cdm)/(ol+cdm) + ((ol)/(ol+cdm))*(1+z)**(3*(w+e)))**(-(e)/(w+e))
    Hz = np.sqrt( ob*(1+z)**3 + cdm*constC*(1+z)**3 +  ol*constC*(1+z)**(3*(1+w+e))) 
    return 1.0 / Hz

def IDEC(cdm,ob,w,e):
    ol = 1 - cdm - ob
    q = np.array([quad(IDEC_Hz_inverse, 0, z, args=(cdm, ol, w, e, ob))[0] for z in zs])
    m = np.array([quad(IDEC_Hz_inverse, 0, z, args=(cdm, ol, w, e, ob))[0] for z in zm])
    y = np.array([quad(IDEC_Hz_inverse, 0, 1090, args=(cdm, ol, w, e, ob))[0]]) # last scattering
    E = y
    F = q
    M = m

    # Calculates values used for the CMB/BAO log likelihoodfor this model
    ang_star = E / (1+1090)
    ang_dist = F / (1 + zs)
    constC = ((cdm)/(ol+cdm) + ((ol)/(ol+cdm))*(1+zs)**(3*(w+e)))**(-(e)/(w+e))
    Hz = np.sqrt( ob*(1+zs)**3 + cdm*constC*(1+zs)**3 +  ol*constC*(1+zs)**(3*(1+w+e))) 
    D_V = ((1 + zs)**2 * ang_dist**2 * (zs)/Hz)**(1/3)
    model = (ang_star)*(1+1090) / D_V
    log_dv = CMB_BAO_log_likelihood(f, f_err, model)

    ang_dist1 =  M / (1 + zm)
    model1 = ((ang_star)*(1+1090)) / ((ang_dist1)*(1+zm))
    log_dm = CMB_BAO_log_likelihood(g, g_err, model1[4:]) # likelihood for dm/rd data

    # calculates the values used for Dh/rd data - zh array of redshifts
    # calculate DH
    constC1 = ((cdm)/(ol+cdm) + ((ol)/(ol+cdm))*(1+zh)**(3*(w+e)))**(-(e)/(w+e))
    Dh = 1/( np.sqrt( ob*(1+zh)**3 + cdm*constC1*(1+zh)**3 +  ol*constC1*(1+zh)**(3*(1+w+e)))  )
    model2 = ((ang_star)*(1+1090)) / Dh
    log_dh = CMB_BAO_log_likelihood(h, h_err, model2[4:]) # likelihood for dh/rd data

    # 4x BOSS
    BOSS_model = np.array([model1[0], model2[0], model1[1], model2[1]])
    log_BOSS = CMB_BAO_cov_log_likelihood(BOSS_model, BOSS_data, BOSS_cov)

    # 2x eBOSS LRG
    LRG_model = np.array([model1[2], model2[2]])
    log_LRG = CMB_BAO_cov_log_likelihood(LRG_model, LRG_data, LRG_cov)

    # 2x eBOSS QU
    QU_model = np.array([model1[3], model2[3]])
    log_QU = CMB_BAO_cov_log_likelihood(QU_model, QU_data, QU_cov)

    # combined likelihood for this specific parameter set against BAO (dv/rd + dm/rd + dh/rd) data
    log_CMB_BAO = log_dm + log_dv + log_dh + log_BOSS + log_LRG + log_QU
    return log_CMB_BAO



if __name__ == "__main__":
    logp = FLCDM(0.27)
    print(logp)