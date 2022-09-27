from telnetlib import DM
import numpy as np
import pandas as pd
from pathlib import Path

# This is a file that contains all of the raw CMB/BAO data. 
# Currently has no output but has the data in the same form used at the top of
# the current likelihood files (that dont use classes)
#TODO: Put into an optimal output that can be used in the likelihood classes.

def get_CMB_BAO_data():
    ######### correlated data
    BOSS_data_path = Path("data/CMB_BAO/BOSS_DDR12_coverted_data.txt")
    BOSS_cov_path = Path("data/CMB_BAO/BOSS_DDR12_coverted_cov.txt")

    BOSS = np.genfromtxt(BOSS_data_path)
    BOSS_cov = np.genfromtxt(BOSS_cov_path)
    BOSS_zz = BOSS[:,0]
    BOSS_data = BOSS[:,1]

    eBOSS_LRG_data_path = Path("data/CMB_BAO/eBOSS_LRG_DDR16_converted_data.txt")
    eBOSS_LRG_cov_path = Path("data/CMB_BAO/eBOSS_LRG_DDR16_converted_cov.txt")
    eBOSS_LRG = np.genfromtxt(eBOSS_LRG_data_path)
    eBOSS_LRG_cov = np.genfromtxt(eBOSS_LRG_cov_path)
    eBOSS_LRG_zz = eBOSS_LRG[:,0]
    eBOSS_LRG_data = eBOSS_LRG[:,1]

    eBOSS_QSO_data_path = Path("data/CMB_BAO/eBOSS_QSO_DDR16_converted_data.txt")
    eBOSS_QSO_cov_path = Path("data/CMB_BAO/eBOSS_QSO_DDR16_converted_cov.txt")

    eBOSS_QSO = np.genfromtxt(eBOSS_QSO_data_path)
    eBOSS_QSO_cov = np.genfromtxt(eBOSS_QSO_cov_path)
    eBOSS_QSO_zz = eBOSS_QSO[:,0]
    eBOSS_QSO_data = eBOSS_QSO[:,1]

    ######### uncorrelated data
    DMonDH = np.genfromtxt('data/CMB_BAO/DM_on_DH_uncorrelated_data.txt', delimiter=',')
    DMonDH_zz = DMonDH[:,0]
    DMonDH_data = DMonDH[:,1]
    DMonDH_err = DMonDH[:,2]
    DMonDM = np.genfromtxt('data/CMB_BAO/DM_on_DM_uncorrelated_data.txt', delimiter=',')
    DMonDM_zz = DMonDM[:,0]
    DMonDM_data = DMonDM[:,1]
    DMonDM_err = DMonDM[:,2]
    DMonDV = np.genfromtxt('data/CMB_BAO/DM_on_DV_uncorrelated_data.txt', delimiter=',')
    DMonDV_zz = DMonDV[:,0]
    DMonDV_data = DMonDV[:,1]
    DMonDV_err = DMonDV[:,2]

if __name__ == "__main__":
   get_CMB_BAO_data()