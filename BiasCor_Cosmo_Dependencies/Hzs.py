import numpy as np
from scipy.integrate import quad
import pandas as pd

# This file grabs the Hz value for each model.

# 1) Flat Cosmological Constant with 1x paramater, \Omega_M - DONE
def FLCDM(z,om):
    ol = 1-om
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return Hz
    
# 2) Cosmological Constant with 2x paramaters, \Omega_M and \Omega_{\Lambda} - DONE
def LCDM(z,om,ol):
    Hz = np.sqrt((1 + z) ** 2 * (om * z + 1) - ol * z * (z + 2))
    return Hz

# 3) Flat Constant wCDM with 2x paramaters, \Omega_M and \omega - DONE
def FwCDM(z,om,w):
    ol = 1 - om
    Hz = np.sqrt((om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return Hz

# 4) Constant wCDM with 3x parameters, \Omega_M, \Omega_{\Lambda} and \omega - DONE
def wCDM(z,om,ol,w):
    omega_k = 1.0 - om - ol
    Hz = np.sqrt((omega_k*(1+z)**(2) + om*(1+z)**(3) + ol*(1+z)**(3*(1+w))))
    return Hz

# 5) Flat w(a) with 3x parameters, \Omega_M, \omega_0 and \omega_a - DONE
def Fwa(z,om,w0,wa):
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0+wa))) * (np.exp(-3*wa*(1-((1+z)**(-1))))) ) )
    return Hz

# 15) IDE1 Q = H e rho_x
def IDE1(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(cdm*(1+z)**3 + ol*(1-(e/(3*w + e)))*(1+z)**((3*(1+w)+e)) + ok*(1+z)**2) 
    return Hz

# 16) IDE2 Q = H e rho_c
def IDE2(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    Hz = np.sqrt(cdm*((1+z)**(3-e)) * (1-(e)/(3*w +e )) + ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
    return Hz

# 18) IDE4 Q = H e [rho_c * rho_x / (rho_c + rho_x)]
def IDE4(z, cdm, ol, w, e):
    ok = 1.0 - cdm - ol
    IDE4_const = ( ((1+z)**(-1*(e+(3*w)))) + (ol/cdm) )**(-e/(e+(3*w)))
    Hz = np.sqrt( (cdm*IDE4_const*(1+z)**(3-e)) + IDE4_const*ol*(1+z)**(3*(1+w)) + ok*(1+z)**2) 
    return Hz

# 19) Flat w(z) with 3x parameters, \Omega_M, \omega_0 and \omega_a - DONE
def Fwz(z,om,w0,wz):
    ol = 1 - om 
    Hz = np.sqrt( (om*(1+z)**(3)) + (ol * ((1+z)**(3*(1+w0-wz))) * (np.exp(3*wz*z)) ) )
    return Hz

# 6) Cardassian with 3x parameters, \Omega_M, q and n
def FCa(z, om, q ,n ):
    Hz = np.sqrt((om*((z+1)**3))*(1+(((om**(-q))-1)*((z+1)**(3*q*(n-1)))))**(1/q))
    return Hz

# 9) Flat General Chaplygin 2x parameters, A and \alpha
def FGChap(z, A, a):
    Hz = np.sqrt((A + (1-A)*((1+z)**(3*(1+a))))**(1.0/(1+a)))
    return Hz

# 8) Chaplygin 2x parameters, A and \Omega_K
def Chap(z, A, ok):
    Hz = np.sqrt(ok*((1+z)**2)+(1-ok)*np.sqrt(A + (1-A)*((1+z)**6)))
    return Hz

# 10) General Chaplygin 3x parameters, \Omega_K, A and \alpha
def GChap(z, A ,a, ok):
    Hz = np.sqrt((ok*((1+z)**2)) + (1-ok)*(A + (1-A)*((1+z)**(3*(1+a))))**(1/(1+a)))
    return Hz

# 11) DGP 2x parameters, \Omega_rc, and \Omega_K
def DGP(z, rc, ok):
    Hz = np.sqrt(ok*((1+z)**2)+ (((np.sqrt(((1 - ok - 2*(np.sqrt(rc)*np.sqrt(1-ok)))*((1+z)**3))+ rc )) + np.sqrt(rc) )**2)) 
    return Hz

# 22) New General Chaplygin 4x parameters, \Omega_K, A and \alpha, w
def NGCG(z, om, A ,a, w):
    Hz = np.sqrt(om*(1+z)**3 + ((1-om)*(1+z)**3)*(1-A*(1-(1+z)**(3*w*(1+a))))**(1/(1+a)))
    return Hz

# 11) Galileon Tracker Solution 2x parameters, \Omega_m, \Omega_g
def GAL(z, om, og):
    ok  = 1 - om - og
    Hz = np.sqrt(0.5*ok*(1+z)**2 + 0.5*om*(1+z)**3 + np.sqrt(og + 0.25*((om*(1+z)+ok)**2)*(1+z)**4))
    return Hz


# 15) IDE1 Q = H e rho_x
def IDEA(z, cdm, w, e):
    ol = 1.0 - cdm
    Hz = np.sqrt(cdm*(1+z)**3 + ol*( ((e)/(w+e))*(1+z)**3 + ((w)/(w+e))*(1+z)**(3*(1+w+e))  )) 
    return Hz

# 16) IDE2 Q = H e rho_c
def IDEB(z, cdm, ob, w, e):
    ol = 1.0 - cdm - ob
    Hz = np.sqrt(ob*(1+z)**(3)+ ol*(1+z)**(3*(1+w)) + cdm*(((e)/(w+e))*(1+z)**(3*(1+w))  + ((w)/(w+e))*(1+z)**(3*(1-e)))) 
    return Hz

# 18) IDE4 Q = H e [rho_c * rho_x / (rho_c + rho_x)]
def IDEC(z, cdm, ob, w, e):
    ol = 1-cdm - ob
    constC = ((cdm)/(ol+cdm) + ((ol)/(ol+cdm))*(1+z)**(3*(w+e)))**(-(e)/(w+e))
    Hz = np.sqrt( ob*(1+z)**3 + cdm*constC*(1+z)**3 +  ol*constC*(1+z)**(3*(1+w+e))) 
    return Hz

def IDEB_2(z, cdm, w, e):
    ol = 1-cdm
    Hz = np.sqrt(ol*(1+z)**(3*(1+w)) + cdm*(((e)/(w+e))*(1+z)**(3*(1+w))  + ((w)/(w+e))*(1+z)**(3*(1-e)))) 
    return Hz

def IDEC_2(z, cdm, w, e):
    ol = 1-cdm
    constC = ((cdm)/(ol+cdm) + ((ol)/(ol+cdm))*(1+z)**(3*(w+e)))**(-(e)/(w+e))
    Hz = np.sqrt( cdm*constC*(1+z)**3 +  ol*constC*(1+z)**(3*(1+w+e))) 
    return Hz


if __name__ == "__main__":
    print(LCDM(1.15838, 0.3, 0.7))
    
