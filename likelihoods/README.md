# The .yaml files here define the default parameters & inputs for the specfic cosmological model

# The class for each model and likelihoods are within non_standards.py

# Mylikes.py & FLCDM_mylikes.yaml are for reference only and should be deleted at some point




# Changes to each model in non_standards.py - Also change default files.

## In initialize add this to everything thats there right now - TAB RIGHT everything currently there
# 1.
        # Load in SN data
        if (self.HD_path != False) & (self.cov_path != False):

# 2.
        if self.CMB_BAO != False:
            x=0 # delete later
            #load in data in an optimal form

## Will also need to change logp..besides the parameters i think everything can be copied over

## function for DM_on_DM, DM_on_DV and DM_on_DH need to be added with input params changed AND the curvature term.