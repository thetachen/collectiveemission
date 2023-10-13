import sys
import numpy as np
from matplotlib import pyplot as plt
from copy import copy
N = 1000
E0 = 1E-6
Wp = 100.0
Wv = 100.0
Gmp = 0.01 * 200
Gmv = 0.01 * 200

gp = 0.5
gv = 0.0

gc_av = 0.2
gc_std = 0.0

fc_std = 0.3
fc_tau = 0.0001

fig1, ax1= plt.subplots(1,4,figsize=(16.0,5.0))
fig2, ax2= plt.subplots(1,2,figsize=(16.0,6.0))

Wmax_list = []
Amax_list = []
Rabi_list = []
FWHM_list = []
# fc_tau_list = np.arange(0.01,2.0,0.2)
fc_tau_list = 10.0**np.arange(-3,2,0.5)
for fc_tau in fc_tau_list:
# if True:
    WW_list = Wv + np.arange(-20.0,20.0,0.02) 
    Avec_list = []
    Ext_list = []
    for WW in WW_list:
        Bmat = np.array([[Wp-WW-1j*Gmp,         gc_av*np.sqrt(N),   fc_std*np.sqrt(N),  0.0                     ],
                        [gc_av*np.sqrt(N),      Wv-WW-1j*Gmv,       0.0,                0.0                     ],
                        [fc_std*np.sqrt(N),     0.0,                Wv-WW-1j*Gmv,       -fc_std*np.sqrt(N)      ],
                        [1j/fc_tau,             0.0,                0.0,                Wv-WW-1j*Gmv-1j/fc_tau  ]],complex)
        fvec = np.array([-gp*E0,    
                        -gv*np.sqrt(N)*E0,
                        0.0,
                        0.0],complex)
        Avec = np.linalg.solve(Bmat,fvec)
        # avec2 = np.dot(np.linalg.inv(Bmat),fvet.T)
        # print(np.allclose(np.dot(Bmat, avec), fvet))
        # print(avec,avec2)
        Avec_list.append(np.abs(Avec)**2)
        Ext_list.append(WW*np.imag(np.dot(fvec,Avec)))
    Avec_list = np.array(Avec_list).T

    ax1[0].plot(WW_list, Avec_list[0], label=r'fc_tau='+str(round(np.log10(fc_tau),2)))
    # ax1[0].plot(WW_list, Avec_list[1], label=r'$|A_s|^2$')
    # ax1[0].plot(WW_list, Avec_list[2], label=r'$|B_c|^2$')
    # ax1[0].plot(WW_list, Avec_list[3], label=r'$|B_f|^2$')
    # ax1[1].plot(WW_list, Ext_list, label=r'fc_tau='+str(fc_tau))

    #find maximum and fwhm:
    idx_lmax = np.argmax(Avec_list[0][:int(len(WW_list)/2):])
    idx_rmax = np.argmax(Avec_list[0][int(len(WW_list)/2):]) + int(len(WW_list)/2)
    Rabi_list.append(WW_list[idx_rmax]-WW_list[idx_lmax])

    Wmax_list.append(WW_list[idx_lmax])
    Amax_list.append(Avec_list[0][idx_lmax])
    Avec_half = Avec_list[0][idx_lmax]/2
    idx_rhalf = idx_lmax
    while Avec_list[0][idx_rhalf] - Avec_half>0:
        idx_rhalf += 1
    idx_lhalf = idx_lmax
    while Avec_list[0][idx_lhalf] - Avec_half>0:
        idx_lhalf -= 1

    FWHM_list.append(WW_list[idx_rhalf]-WW_list[idx_lhalf])


    #analytical inverse:
    Aana_list = []
    Aana_numerator = []
    Aana_denominator = []
    for WW in WW_list:
        p = Wp-WW-1j*Gmp
        v = Wv-WW-1j*Gmv
        c = gc_av*np.sqrt(N)
        f = fc_std*np.sqrt(N)
        y = 1j/fc_tau
        Aana_list.append( np.abs( -gp*E0 * v*(y-v)/(c**2*(v-y)+ v*(f**2+p*(y-v))) )**2 )
        Aana_denominator.append( np.abs( 1.0/(c**2*(v-y)+ v*(f**2+p*(y-v))) )**2 )
        Aana_numerator.append( np.abs( -gp*E0 * v*(y-v) )**2 )
    # ax1[0].plot(WW_list, Aana_list, '-x')
    ax2[0].plot(WW_list, Aana_numerator)
    ax2[1].plot(WW_list, Aana_denominator)


ax1[0].legend()
ax1[0].set_xlabel('Driving frequency (cm$^{-1}$)')
ax1[0].set_ylabel('Transmission')
# ax1[0].set_title('gp={gp}; gv={gv}; gc_av={gc_av}, gc_std={gc_std}; fc_std={fc_std}, fc_tau={fc_tau}'.format(**locals()))

# label the maximum
ax1[0].scatter(Wmax_list, Amax_list)

ax1[1].legend()
plt.tight_layout()
ax1[1].semilogx(fc_tau_list,Amax_list)
ax1[2].semilogx(fc_tau_list,Rabi_list)
ax1[3].semilogx(fc_tau_list,FWHM_list)
ax1[1].set_xlabel(r'$\tau_f$')
ax1[1].set_ylabel('Amax')
ax1[2].set_xlabel(r'$\tau_f$')
ax1[2].set_ylabel('Rabi splitting')
ax1[3].set_xlabel(r'$\tau_f$')
ax1[3].set_ylabel('FWHM')
plt.show()