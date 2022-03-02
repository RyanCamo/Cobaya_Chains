# Cobaya_Chains
Repository for computing MCMC chains for non-standard models using Cobaya.

The output chains are saved into the 'chains' folder - which have various sub folders depending on which data set we are using to constrain a model
ie. SN, CMB/BAO, SN+CMB/BAO

The 'data' folder containts supernova data including: DES5YR_UNBIN, DES5YR_BIN, DES3YR_UNBIN, DES3YR_BIN. Where the DES5YR are both mocks.

The .yaml files represent jobs to be completed much like pippin. 
The current .yaml files here are jobs that have already been run and can be used as examples or changed as required.

Each .yaml file will have a line of the form:
  external: import_module('X').Y

BEFORE running a particular job:

'X' can be changed to:

    likelihoods_BAO_CMB     - To constrain the model Y against CMB/BAO data
  
    likelihoods_SN          - To constrain the model Y against SN data
  
    likelihoods_BAO_CMB_SN  - To constrain model Y against SN+CMB/BAO data
    
    By default the SN data is using DES5YR_UNBIN. 
    This can be changed within the likelihood files that constrain against SN data by changing the relative file paths. 
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


Finally, make sure to change the name of the output folder at the bottom of the .yaml file.


To run a job:

Type 'cobaya-run Z.yaml' in the terminal. Where Z is the name of your .yaml file.
