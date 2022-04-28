import numpy as np
import sys
sys.path.append('Cobaya_Chains')
from matplotlib import pyplot as plt
from chainconsumer import ChainConsumer
import uncertainties
from uncertainties import ufloat
c = ChainConsumer()


# This document uses raw data from Planck & eBOSS (Alam et al. 2020) and converts these measurements
# into the form used in the thesis. It then prints of the rounded table used in the CMB/BAO 
# constraining section and prints the data in a form I used to get fits and create contours

#####################################
# Data from planck:
theta_star = 0.0104109
theta_star_err = 0.000003
ratio = 1.018 # ratio = rd/r_*
ratio_err = 0.0030
theta_star1 = ufloat(0.0104109, 0.000003)
ratio1 = ufloat(1.018, 0.0030)
#####################################


# There is a total of 14 measurements presented in eBOSS and used in this thesis. The majority are 
# provided at: https://svn.sdss.org/public/data/eboss/DR16cosmo/tags/v1_0_1/likelihoods/BAO-only/
# The MGS measurement is 
# provided at: https://academic.oup.com/mnras/article/449/1/835/1298372#supplementary-data

#####################################

#  4x BAO Only measurements from BOSS (DR12)
# This section calculates the measurments and propogates the covariance matrix between

DMDH0 = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR12_LRG_BAO_DMDH.txt', comments='#') # From eBOSS
DMDH= DMDH0[:,1]
DMDH_cov0 = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR12_LRG_BAO_DMDH_covtot.txt', comments='#') # From eBOSS
DMDH_cov1 = DMDH_cov0.reshape(4,4) 
DMDH_with_cov = uncertainties.correlated_values(DMDH, DMDH_cov1)  # Tell Uncertainties the associated covariance matrix
X1 = (1/theta_star1 * 1/DMDH_with_cov[0])/ ratio1 
X2 = (1/theta_star1 * 1/DMDH_with_cov[1])/ ratio1
X3 = (1/theta_star1 * 1/DMDH_with_cov[2])/ ratio1
X4 = (1/theta_star1 * 1/DMDH_with_cov[3])/ ratio1
BOSS_values_error = np.array(uncertainties.covariance_matrix([X1, X2, X3, X4])) # Outputs the covariance matrix
BOSS_values = np.array([X1.nominal_value, X2.nominal_value, X3.nominal_value, X4.nominal_value])

test = ufloat(1,0.1)
test1 = ufloat(2,0.2)
test2 = np.array(uncertainties.covariance_matrix([test, test1]))


#####################################

# 1x MGS measurement # z_eff = 0.15
# r_d^{\rm fid} = 148.69 Mpc.
#D^{\rm fid}_V(z) = 638.95$ Mpc
MGS_DV_over_rs = 4.46566682359271
MGS_DV_over_rs_err = 0.16813504606900265

# Conversion
MGS_value = (1/theta_star * 1/MGS_DV_over_rs)/ ratio
# Error propogation:
MGS_value_error = MGS_value * np.sqrt((theta_star_err/theta_star)**2 + (MGS_DV_over_rs_err/MGS_DV_over_rs)**2 + (ratio_err/ratio)**2)

#####################################

# 1x ELG Measurement - z_eff = 0.85
# First load in the samples/likelihoods
ELG_DV_over_rs_sample = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR16_ELG_BAO_DVtable.txt', comments='#', usecols=(0))
ELG_DV_over_rs_likelihood = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR16_ELG_BAO_DVtable.txt', comments='#', usecols=(1))

# Get Results from ChainConsumer
test = c.add_chain(ELG_DV_over_rs_sample, weights=ELG_DV_over_rs_likelihood, grid=True, name="MCMC", statistics='max')
c.configure(smooth=1) # Data was smoothed in the eBoss paper - I did minor smoothing
ELG_DV_over_rs  = c.analysis.get_summary(chains="MCMC")['0'][1]
ELG_DV_over_rs_up  = c.analysis.get_summary(chains="MCMC")['0'][2] - c.analysis.get_summary(chains="MCMC")['0'][1]
ELG_DV_over_rs_low  = c.analysis.get_summary(chains="MCMC")['0'][1] - c.analysis.get_summary(chains="MCMC")['0'][0]
ELG_DV_over_rs_err = np.max(np.array([ELG_DV_over_rs_up, ELG_DV_over_rs_low]))

# Following the method in the thesis (Eqs 4.20 & 4.22):
ELG_DV_values = (1/theta_star * 1/ELG_DV_over_rs)/ ratio

# Error propogation:
ELG_DV_values_error = ELG_DV_values * np.sqrt((theta_star_err/theta_star)**2 + (ELG_DV_over_rs_err/ELG_DV_over_rs)**2 + (ratio_err/ratio)**2)
c.remove_chain("MCMC")

#####################################

#  2x BAO Only measurements from eBOSS LRG (DR16) z_eff = 0.698
LRG_DMDH_over_rs_0 = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR16_LRG_BAO_DMDH.txt', comments='#')
LRG_DMDH_over_rs= LRG_DMDH_over_rs_0[:,1]
LRG_DMDH_over_cov_0 = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR16_LRG_BAO_DMDH_covtot.txt', comments='#')
LRG_DMDH_over_cov = LRG_DMDH_over_cov_0.reshape(2,2) 
LRG_DMDH_over_rs_with_cov = uncertainties.correlated_values(LRG_DMDH_over_rs, LRG_DMDH_over_cov)
XX1 = (1/theta_star1 * 1/LRG_DMDH_over_rs_with_cov[0])/ ratio1
XX2 = (1/theta_star1 * 1/LRG_DMDH_over_rs_with_cov[1])/ ratio1
eBOSS_LRG_values_error = np.array(uncertainties.covariance_matrix([XX1, XX2]))
eBOSS_LRG_values = np.array([XX1.nominal_value, XX2.nominal_value])


#####################################

#  2x BAO Only measurements from eBOSS Quasar (DR16) z_eff = 1.48
QU_DMDH_over_rs_0 = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR16_QSO_BAO_DMDH.txt', comments='#')
QU_DMDH_over_rs = QU_DMDH_over_rs_0[:,1]
QU_DMDH_over_cov_0 = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR16_QSO_BAO_DMDH_covtot.txt', comments='#')
QU_DMDH_over_cov = QU_DMDH_over_cov_0.reshape(2,2) 
QU_DMDH_over_rs_with_cov = uncertainties.correlated_values(QU_DMDH_over_rs, QU_DMDH_over_cov)
XXX1 = (1/theta_star1 * 1/QU_DMDH_over_rs_with_cov[0])/ ratio1
XXX2 = (1/theta_star1 * 1/QU_DMDH_over_rs_with_cov[1])/ ratio1
eBOSS_QU_values_error = np.array(uncertainties.covariance_matrix([XXX1, XXX2]))
eBOSS_QU_values = np.array([XXX1.nominal_value, XXX2.nominal_value])

#####################################

# 2x LYAUTO Measurement z_eff = 2.334
# First load in the samples/likelihoods
LYAUTO_DMDH_over_rs_sample = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR16_LYAUTO_BAO_DMDHgrid.txt', comments='#', usecols=(0,1))
LYAUTO_DMDH_over_rs_likelihood = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR16_LYAUTO_BAO_DMDHgrid.txt', comments='#', usecols=(2))

# Get Results from ChainConsumer
test = c.add_chain(LYAUTO_DMDH_over_rs_sample, weights=LYAUTO_DMDH_over_rs_likelihood, grid=True, name="MCMC", statistics='max')
c.configure(smooth=1) # Data was smoothed in the eBoss paper - I did minor smoothing
LYAUTO_DMDH_over_rs  = c.analysis.get_summary(chains="MCMC")
LYAUTO_DM_over_rs_up  = c.analysis.get_summary(chains="MCMC")['0'][2] - c.analysis.get_summary(chains="MCMC")['0'][1]
LYAUTO_DM_over_rs_low  = c.analysis.get_summary(chains="MCMC")['0'][1] - c.analysis.get_summary(chains="MCMC")['0'][0]
LYAUTO_DH_over_rs_up  = c.analysis.get_summary(chains="MCMC")['1'][2] - c.analysis.get_summary(chains="MCMC")['1'][1]
LYAUTO_DH_over_rs_low  = c.analysis.get_summary(chains="MCMC")['1'][1] - c.analysis.get_summary(chains="MCMC")['1'][0]

# The measurements extracted from the likelihoods are:
LYAUTO_DMDH_over_rs = np.array([c.analysis.get_summary(chains="MCMC")['0'][1],c.analysis.get_summary(chains="MCMC")['1'][1]])
LYAUTO_DMDH_over_rs_err = np.array([np.max(np.array([LYAUTO_DM_over_rs_up, LYAUTO_DM_over_rs_low])), np.max(np.array([LYAUTO_DH_over_rs_up, LYAUTO_DH_over_rs_low]))])

# Following the method in the thesis (Eqs 4.20 & 4.22):
eBOSS_LYAUTO_values = (1/theta_star * 1/LYAUTO_DMDH_over_rs)/ ratio
print(eBOSS_LYAUTO_values)

# Error propogation:
eBOSS_LYAUTO_values_error = eBOSS_LYAUTO_values * np.sqrt((theta_star_err/theta_star)**2 + (LYAUTO_DMDH_over_rs_err/LYAUTO_DMDH_over_rs)**2 + (ratio_err/ratio)**2)
print(eBOSS_LYAUTO_values_error)
c.remove_chain("MCMC")

#####################################

# 2x LYxQU Measurement z_eff = 2.334
# First load in the samples/likelihoods
LYxQU_DMDH_over_rs_sample = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR16_LYxQSO_BAO_DMDHgrid.txt', comments='#', usecols=(0,1))
LYxQU_DMDH_over_rs_likelihood = np.loadtxt('Cobaya_Chains/CMB_BAO_Data/eBOSS/sdss_DR16_LYxQSO_BAO_DMDHgrid.txt', comments='#', usecols=(2))

# Get Results from ChainConsumer
test = c.add_chain(LYxQU_DMDH_over_rs_sample, weights=LYxQU_DMDH_over_rs_likelihood, grid=True, name="MCMC", statistics='max')
c.configure(smooth=1) # Data was smoothed in the eBoss paper - I did minor smoothing
LYxQU_DMDH_over_rs  = c.analysis.get_summary(chains="MCMC")
LYxQU_DM_over_rs_up  = c.analysis.get_summary(chains="MCMC")['0'][2] - c.analysis.get_summary(chains="MCMC")['0'][1]
LYxQU_DM_over_rs_low  = c.analysis.get_summary(chains="MCMC")['0'][1] - c.analysis.get_summary(chains="MCMC")['0'][0]
LYxQU_DH_over_rs_up  = c.analysis.get_summary(chains="MCMC")['1'][2] - c.analysis.get_summary(chains="MCMC")['1'][1]
LYxQU_DH_over_rs_low  = c.analysis.get_summary(chains="MCMC")['1'][1] - c.analysis.get_summary(chains="MCMC")['1'][0]

# The measurements extracted from the likelihoods are:
LYxQU_DMDH_over_rs = np.array([c.analysis.get_summary(chains="MCMC")['0'][1],c.analysis.get_summary(chains="MCMC")['1'][1]])
LYxQU_DMDH_over_rs_err = np.array([np.max(np.array([LYxQU_DM_over_rs_up, LYxQU_DM_over_rs_low])), np.max(np.array([LYxQU_DH_over_rs_up, LYxQU_DH_over_rs_low]))])

# Following the method in the thesis (Eqs 4.20 & 4.22):
eBOSS_LYxQU_values = (1/theta_star * 1/LYxQU_DMDH_over_rs)/ ratio
#print(eBOSS_LYxQU_values)

# Error propogation:
eBOSS_LYxQU_values_error = eBOSS_LYxQU_values * np.sqrt((theta_star_err/theta_star)**2 + (LYxQU_DMDH_over_rs_err/LYxQU_DMDH_over_rs)**2 + (ratio_err/ratio)**2)
#print(eBOSS_LYxQU_values_error)
c.remove_chain("MCMC")

#####################################
# Print the results in a form used I used specifically for CMB/BAO MCMC fitting. 

## DM/DV data - no need to worry about correlations 
print('### DM/DV Data:')
print('zs = np.array([0.15, 0.845])')
print('f = np.array([%s, %s])' % (MGS_value, ELG_DV_values))
print('f_err = np.array([%s, %s])' % (MGS_value_error, ELG_DV_values_error))
print('### Redshifts for the rest of the measurements:')
print('zm = np.array([0.38, 0.51, 0.698, 1.48, 2.334, 2.334])')
print('zh = np.array([0.38, 0.51, 0.698, 1.48, 2.334, 2.334])')
print('### Uncorrelated DM/DM data:')
print('g = np.array([%s, %s])' %(eBOSS_LYAUTO_values[0],eBOSS_LYxQU_values[0]))
print('g_err = np.array([%s, %s])'%(eBOSS_LYAUTO_values_error[0],eBOSS_LYxQU_values_error[0] ))
print('### Uncorrelated DM/DH data:')
print('h = np.array([%s, %s])' %(eBOSS_LYAUTO_values[1],eBOSS_LYxQU_values[1]))
print('h_err = np.array([%s, %s])'%(eBOSS_LYAUTO_values_error[1],eBOSS_LYxQU_values_error[1] ))
print('### Correlated data for BOSS:')
print('BOSS_data = np.array([%s, %s, %s, %s])' %(BOSS_values[0],BOSS_values[1],BOSS_values[2],BOSS_values[3]))
print('BOSS_cov = np.array(%s)' %(BOSS_values_error))
print('### Correlated data for LRG:')
print('LRG_data = np.array([%s, %s])' %(eBOSS_LRG_values[0],eBOSS_LRG_values[1]))
print('LRG_cov = np.array(%s)' %(eBOSS_LRG_values_error))
print('### Correlated data for QU:')
print('QU_data = np.array([%s, %s,])' %(eBOSS_QU_values[0],eBOSS_QU_values[1]))
print('QU_cov = np.array(%s)' %(eBOSS_QU_values_error))

#####################################
# Print the table used to summarise these results:

print(r'\begin{table}[h!]')
print(r'\centering')
print(r'\begin{threeparttable}')
print(r'\caption{\textsc{Combined CMB and BAO Measurements}}')
print(r'\label{tab:CMBBAOdata}')
print(r'\begin{tabular*}{\textwidth}{c @{\extracolsep{\fill}} ccc}') 
print(r'\hline\hline') 
print(r'$z_{\text{eff}}$   &  $D_M(z_*)/D_V(z)$ & $D_M(z_*)/D_M(z)$& $D_M(z_*)/D_H(z)$\Tstrut\Bstrut\\') 
print(r'\hline')
print(r'0.15 & $%s \pm %s$ & - & -  \Tstrut\Bstrut\\' %(np.round(MGS_value,2), np.round(MGS_value_error,2) ))  # 2 to be added
print(r'0.38 & - & $%s \pm %s$ & $%s \pm %s$\Tstrut\Bstrut\\' %(np.round(BOSS_values[0],2), np.round(np.sqrt(BOSS_values_error[0,0]),2), np.round(BOSS_values[1],2), np.round(np.sqrt(BOSS_values_error[1,1]),2) ))
print(r'0.51 & - & $%s \pm %s$ & $%s \pm %s$\Tstrut\Bstrut\\' %(np.round(BOSS_values[2],2), np.round(np.sqrt(BOSS_values_error[2,2]),2), np.round(BOSS_values[3],2), np.round(np.sqrt(BOSS_values_error[3,3]),2) )) 
print(r'0.70 & - & $%s \pm %s$ & $%s \pm %s$\Tstrut\Bstrut\\' %(np.round(eBOSS_LRG_values[0],2), np.round(np.sqrt(eBOSS_LRG_values_error[0,0]),2), np.round(eBOSS_LRG_values[1],2), np.round(np.sqrt(eBOSS_LRG_values_error[1,1]),2) )) 
print(r'0.85 & $%s \pm %s$ & - & - \Tstrut\Bstrut\\' %(np.round(ELG_DV_values,2), np.round(ELG_DV_values_error,2)))  
print(r'1.48 & - & $%s \pm %s$ & $%s \pm %s$\Tstrut\Bstrut\\' %(np.round(eBOSS_QU_values[0],2), np.round(np.sqrt(eBOSS_QU_values_error[0,0]),2),np.round(eBOSS_QU_values[1],2),np.round(np.sqrt(eBOSS_QU_values_error[1,1]),2) ))  
print(r'2.33 & - & $%s \pm %s$ & $%s \pm %s$\Tstrut\Bstrut\\' %(np.round(eBOSS_LYAUTO_values[0],2),np.round(eBOSS_LYAUTO_values_error[0],2),np.round(eBOSS_LYAUTO_values[1],2),np.round(eBOSS_LYAUTO_values_error[1],2)))  
print(r'2.33 & - & $%s \pm %s$ & $%s \pm %s$\Tstrut\Bstrut\\' %(np.round(eBOSS_LYxQU_values[0],2),np.round(eBOSS_LYxQU_values_error[0],2),np.round(eBOSS_LYxQU_values[1],2),np.round(eBOSS_LYxQU_values_error[1],2)))  
print(r'\hline')
print(r'\end{tabular*}')
print(r'\end{threeparttable}')
print(r'\end{table}')