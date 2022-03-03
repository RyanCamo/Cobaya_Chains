import scipy.special as sc

chi2 =207.4
dof = 191

Gamma1 = sc.gammaincc(0.5*dof, 0.5*chi2)
print(Gamma1)
Gamma2 = sc.gamma(0.5*dof)
#print(Gamma2)

#scipy.special.gammainc(0.5*288)
GoF = Gamma1*Gamma2
#print(GoF)