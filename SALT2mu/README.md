## QUICK NOTES ##

https://github.com/RyanCamo/Cobaya_Chains

The relevant files to mess with SALT2mu are:
	a) SALT2mu/likelihood.py
	b) SALT2mu/SALT2mu.yaml
	c) test.yaml
	d) nonstandard.FITRES

a) SALT2mu/likelihood.py

This file has the implementation of the SALT2mu log likelihood. Most of the code I have implemented is simply Eq. 3 from Ricks 2017 paper: https://arxiv.org/abs/1610.04677. I have added the approximation of the G10 intrinsic scatter model in sigmamu2 (defined in a comment below)

b) SALT2mu/SALT2mu.yaml

This file has the default inputs/priors. Can adjust the Fiducial cosmology here.


c) test.yaml

This is the file that we use to run our job. To run the cobaya job, you require cobaya and from your shell 'cobaya-run test.yaml'

d) nonstandard.FITRES

This is the output file from BBC im trying to mimic. This was created using a Fiducial cosmology:
w=-1.5, om = 0.5, ol = 0.5. I use the BiasCor values from this output file as if I got them from the interpolator.
HOWEVER, im not grabbing the correct biascor alpha/beta values as I sample alpha and beta.