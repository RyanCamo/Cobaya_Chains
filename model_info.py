# Getinfo function
def get_info(x, *params):
    if x == 'FLCDM':
        label = [r"$\Omega_m$"]
        begin = [0.3]
        if len(params) > 0:
            legend = r'$F\Lambda$: $\Omega_m = %0.2f $' % (params[0])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'LCDM':
        label = [r"$\Omega_m$",r"$\Omega_{\Lambda}$"]
        begin = [0.3, 0.7]
        if len(params) > 0:
            legend = r'$\Lambda$: $\Omega_m = %0.2f $, $\Omega_{\Lambda} = %0.2f $' % (params[0], params[1] )
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'FwCDM':
        label = [r"$\Omega_m$",r"$\omega$"]
        begin = [0.3, -1]
        if len(params) > 0:
            legend = r'F$\omega$: $\Omega_m = %0.2f $, $\omega = %0.2f $' % (params[0], params[1] )
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'wCDM':
        label = [r"$\Omega_m$",r"$\Omega_{\Lambda}$",r"$\omega$"]
        begin = [0.3, 0.7, -1]
        if len(params) > 0:
            legend = r'$\omega$: $\Omega_m = %0.2f $, $\Omega_{\Lambda} = %0.2f $, $\omega = %0.2f $' % (params[0], params[1], params[2] )
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'Fwa':
        label = [r"$\Omega_m$",r"$w_0$",r"$w_a$"]
        begin = [0.3, -1.1, 0.8]
        if len(params) > 0:
            legend = r'F$\omega$(a): $\Omega_m = %0.2f $, $\omega_0 = %0.2f $, $\omega_a = %0.2f $' % (params[0], params[1], params[2] )
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'Fwz':
        label = [r"$\Omega_m$",r"$w_0$",r"$w_z$"]
        begin = [0.3, -1.1, 1.25]
        if len(params) > 0:
            legend = r'F$\omega$(z): $\Omega_m = %0.2f $, $\omega_0 = %0.2f $, $\omega_z = %0.2f $' % (params[0], params[1], params[2] )
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'FCa':
        label = [r"$\Omega_m$",r"$q$","$n$"]
        begin = [0.3, 1, 0.01]
        if len(params) > 0:
            legend = r'FCa: $\Omega_m = %0.2f $, $q = %0.2f $,$n = %0.2f $' % (params[0], params[1], params[2] )
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 7:
        label = ["A"]
        begin = [0] 
        if len(params) > 0:
            legend = 'Model not used'
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'Chap':
        label = [r"$A$",r"$\Omega_K$"]
        begin = [0.8, 0.2]
        if len(params) > 0:
            legend = r'SCh: $A = %0.2f $ $\Omega_K = %0.2f $' % (params[0], params[1] )
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'FGChap':
        label = [r"$A$", r"$\alpha$"]
        begin = [0.7, 0.2]
        if len(params) > 0:
            legend = r'FGCh: $A = %0.2f $ $\alpha = %0.2f $' % (params[0], params[1])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'GChap':
        label = [r"$A$",r"$\alpha$",r"$\Omega_K$"]
        begin = [0.7, 0.3,0.01]
        if len(params) > 0:
            legend = r'GCh: $A = %0.2f $ $\alpha = %0.2f $ $\Omega_K = %0.2f $' % (params[0], params[1], params[2])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'DGP':
        label = [r"$\Omega_{rc}$", r"$\Omega_{K}$"]
        begin = [0.13, 0.02]
        if len(params) > 0:
            legend = r'DGP: $\Omega_{rc} = %0.2f $, $\Omega_{K} = %0.2f $' % (params[0], params[1])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 12:
        label = [r"\Omega_{rc}"]
        begin = [0]
        if len(params) > 0:
            legend = 'Model not used'
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'IDE0':
        label = [r"$\Omega_{CDM}$", r"$\Omega_{DE}$", r"$\omega$", r"$Q$"]
        begin = [0.3, 0.7, -1, -0.02]
        if len(params) > 0:
            legend = r'IDE: $\Omega_{CDM} = %0.2f $, $\Omega_{DE} = %0.2f $, $\omega = %0.2f $, $Q = %0.2f $' % (params[0], params[1], params[2], params[3])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'IDE1':
        label = [r"$\Omega_{m}$", r"$\Omega_{x}$", r"$\omega$", r"$\varepsilon$"]
        begin = [0.25, 0.75, -1.1, 0.05]
        if len(params) > 0:
            legend = r'IDE 1: $\Omega_{CDM} = %0.2f $, $\Omega_{DE} = %0.2f $, $\omega = %0.2f $, $\epsilon = %0.2f $' % (params[0], params[1], params[2], params[3])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'IDE2':
        label = [r"$\Omega_{m}$", r"$\Omega_{x}$", r"$\omega$", r"$\varepsilon$"]
        begin = [0.25, 0.75, -0.9, 0.03]
        if len(params) > 0:
            legend = r'IDE 2: $\Omega_{CDM} = %0.2f $, $\Omega_{DE} = %0.2f $, $\omega = %0.2f $, $\epsilon = %0.2f $' % (params[0], params[1], params[2], params[3])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'IDE4':
        label = [r"$\Omega_{m}$", r"$\Omega_{x}$", r"$\omega$", r"$\varepsilon$"]
        begin = [0.25, 0.75, -0.9, 0.03]
        if len(params) > 0:
            legend = r'IDE 3: $\Omega_{CDM} = %0.2f $, $\Omega_{DE} = %0.2f $, $\omega = %0.2f $, $\epsilon = %0.2f $' % (params[0], params[1], params[2], params[3])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'LTBg':
        label = [r"$\Omega_{out}$", r"$\Omega_{in}$", r"$z_0$"]
        begin = [0.35, 0.25, 0.1]
        if len(params) > 0:
            legend = r'LTBg: $\Omega_{out} = %0.2f $, $\Omega_{in} = %0.2f $, $z_0 = %0.2f $' % (params[0], params[1], params[2])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'GLT':
        label = [r"$\Omega_{m}}$",r"$z_t$"]
        begin = [0.3, 3]
        if len(params) > 0:
            legend = r'GLT: $\Omega_{m} = %0.2f $, $z_t = %0.2f $' % (params[0], params[1])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'NGCG':
        label = [r"$\Omega_m$", r"$A$",r"$\alpha$",r"$\omega$"]
        begin = [0.3, 0.5, 0.2, -0.8]
        if len(params) > 0:
            legend = r'NGCG: $\Omega_{m} = %0.2f $, $A = %0.2f $, $\alpha = %0.2f$, $\omega = %0.2f$' % (params[0], params[1],params[2], params[3])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'GAL':
        label = [r"$\Omega_m$", r"$\Omega_g$" ]
        begin = [0.3, 0.6]
        if len(params) > 0:
            legend = r'GAL: $\Omega_{m} = %0.2f $, $\Omega_g = %0.2f$' % (params[0], params[1])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'IDEA':
        label = [r"$\Omega_{m}$", r"$\omega$", r"$\varepsilon$"]
        begin = [0.25, -1.1, 0.05]
        if len(params) > 0:
            legend = r'IDE 1: $\Omega_{CDM} = %0.2f $,  $\omega = %0.2f $, $\epsilon = %0.2f $' % (params[0], params[1], params[2])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'IDEB':
        label = [r"$\Omega_{CDM}$", r"$\Omega_{b}$", r"$\omega$", r"$\varepsilon$"]
        begin = [0.25, 0.05, -0.9, 0.03]
        if len(params) > 0:
            legend = r'IDE 2: $\Omega_{CDM} = %0.2f $, $\Omega_{b} = %0.2f $, $\omega = %0.2f $, $\epsilon = %0.2f $' % (params[0], params[1], params[2], params[3])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'IDEB_2':
        label = [r"$\Omega_{CDM}$", r"$\omega$", r"$\varepsilon$"]
        begin = [0.25, -0.9, 0.03]
        if len(params) > 0:
            legend = r'IDE 2: $\Omega_{CDM} = %0.2f $, $\omega = %0.2f $, $\epsilon = %0.2f $' % (params[0], params[1], params[2])
        else:
            legend = 'No parameters provided'
        return label, begin, legend

    if x == 'IDEC':
        label = [r"$\Omega_{m}$", r"$\Omega_{b}$", r"$\omega$", r"$\varepsilon$"]
        begin = [0.25, 0.05, -0.9, 0.03]
        if len(params) > 0:
            legend = r'IDE 3: $\Omega_{CDM} = %0.2f $, $\Omega_{DE} = %0.2f $, $\omega = %0.2f $, $\epsilon = %0.2f $' % (params[0], params[1], params[2], params[3])
        else:
            legend = 'No parameters provided'
        return label, begin, legend