# Cobaya_Chains
Repository for computing MCMC chains for non-standard models using Cobaya.

The output chains are saved into the 'chains' folder - which have various sub folders depending on which data set we are using to constrain a model
ie. SN, CMB/BAO, SN+CMB/BAO

# Data
The 'data' folder containts supernova data.
The data sets used in my thesis are:
UNBIN_DES5YR_LOWZ_data.txt & UNBIN_DES5YR_LOWZ_cov.txt - Mock Sample
DES5YR_REAL_DIFFIMG_DATA.txt & DES5YR_REAL_DIFFIMG_COV.txt - REAL Blinded Sample (Prior to BiasCor Corrections)

# Result Generators
The Result Generators folder contains various plots created for my thesis (HubbleDiagram.py, RelativeChanges_avg_4x.py)
as well as the code used to get the best fit parameters/GoF%/chi^2 per DoF etc from the chains - note that some of these
files have further instructions.

# Contours
The contours folder contains all the code for the contour plots and the output images. There is another README.md file in this folder

# CMB_BAO_Data
This folder contains the data from eBOSS in a subdirectory. The BAO_conversions.py file converts the measurements from eBOSS and Planck into the form used within this thesis. 

# BiasCor_Csomo_Dependencies
This folder contains:
Hz.py                     - returns H(z) for each model
HzFUN_FILE_Generator.py   - creates a file with 2 colums: z H(z) using the best fit parameters the first SN analysis
                            this file is saved to Files_2 and used within pippin to generate BiasCor simulations for the second analysis.
PIPPIN_OUTPUTS            - The resulting Hubble diagram after using the new BiasCor simulations for each model
                            (referenced as'MOD')
residual_plot.py          - compares the original Hubble diagram to the new Hubble diagram (figures in thesis)


# YAML FILES

The .yaml files represent jobs to be completed much like pippin. 
The current .yaml files here are jobs that have already been run and can be used as examples or changed as required.

Each .yaml file will have a line of the form:
  external: import_module('X').Y

BEFORE running a particular job:

'X' can be changed to:

    likelihoods_BAO_CMB     - To constrain the model Y against CMB/BAO data
  
    likelihoods_SN          - To constrain the model Y against SN data (Iteration 1)

    likelihoods_SN_bias     - To constrain the model Y against SN data (Iteration 2) - Need to change line 19 of 
                              likelihoods_SN_bias.py to the model your looking at
  
    likelihoods_BAO_CMB_SN  - To constrain model Y against SN+CMB/BAO data
    
    By default the SN data is using DES5YR_UNBIN. 
    This can be changed within the likelihood files that constrain models against SN data by changing the relative file paths. 
    Note however that different SN data need to be imported differently.
  
  
  Y represents the abbreviation for a particular non standard model. 
  
  The non-standard models currently supported are:
  
    FLCDM   - Flat Lambda_CDM
    
    LCDM    - Lambda_CDM
    
    FwCDM   - Flat omega_CDM
    
    wCDM    - omega_CDM
    
    Fwa     - Flat linear parameterisation of omega as a function of scalefactor
    
    Fwz     - Flat linear parameterisation of omega as a function of redshift
    
    IDE1    - Interacting Dark Energy & Dark Matter: Q = Hερ_d
    
    IDE2    - Interacting Dark Energy & Dark Matter: Q = Hερ_c
    
    IDE4    - Interacting Dark Energy & Dark Matter: Q = Hε (ρ_c * ρ_d)/(ρ_c + ρ_d)
    
    Chap    - Standard Chaplygin Gas
    
    FGChap  - Flat Generalised Chaplygin Gas
    
    GChap   - Generalised Chaplygin Gas
    
    NGCG    - New Generalised Chaplygin Gas
    
    FCa     - Flat Cardassian 
    
    DGP     - DGP 


Finally, make sure to change the name of the output file at the bottom of the .yaml and the file path depending on where you want your output saved.


To run a job:

Type 'cobaya-run Z.yaml' in the terminal. Where Z is the name of your .yaml file.
