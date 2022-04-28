This folder contains code that creates tables for relevent sections of the analysis.

## model comparisons.py 

This file creates a table that compares each non-standard model. It displays: chi^2/dof, GoF (%), AIC & BIC. 

Note: To run this code, by default the contours for each model need to be created first (which saves the relevant burn-in for those specific contours & best fit parameters). This however can be overwritten by uncommenting   "#burn = 0 # Uncomment to overwrite" at line 74 (at the time this is written).
