import numpy as np
from os.path import isfile
from matplotlib import pyplot as plt
from functions import get_metadata, get_grid, read_params, get_az_data

data_loc = '/home/cwp29/diablo3/strat/high_res/verif/as_data.npy'

if isfile(data_loc):
    data_array = np.load(data_loc, allow_pickle=False)
else:
    print("ERROR: Data file not found.")

z_fig, z_ax = plt.subplots(1,4, figsize=(14, 4), constrained_layout=True)
#v_fig, v_ax = plt.subplots(1,2, figsize=(9, 4), constrained_layout=True)

markers = {
        0.2 : 'o',
        0.15 : '^',
        0.1 : 's',
        0.05 : '*'
        }

xs = np.linspace(0, 30, 5)
ys = np.linspace(0, 30, 5)
z_ax[0].plot(xs, ys, linestyle='--', color='k', alpha=0.5)
z_ax[1].plot(xs, ys, linestyle='--', color='k', alpha=0.5)
z_ax[3].plot(xs, 0.12*xs, linestyle='--', color='k', alpha=0.5)

for i in range(len(data_array)):
    md = get_metadata(data_array[i,0], "3.7")

    # Zmax
    z_ax[0].scatter(100*(float(data_array[i,1])-md['H']), 100*(float(data_array[i,2])-md['H']),
            marker=markers[md['H']], color='b')
    z_ax[0].scatter(100*(float(data_array[i,10])), 100*(float(data_array[i,2])-md['H']),
            marker=markers[md['H']], color='r')

    # Zeq
    z_ax[1].scatter(100*(float(data_array[i,3])-md['H']), 100*(float(data_array[i,4])-md['H']),
            marker=markers[md['H']], color='b')
    z_ax[1].scatter(100*(float(data_array[i,11])), 100*(float(data_array[i,4])-md['H']),
            marker=markers[md['H']], color='r')

    # zss ratio vs. fr
    z_ax[2].scatter(float(data_array[i,6]), (float(data_array[i,5])-md['H'])/(float(data_array[i,2])-md['H']),
            marker=markers[md['H']], color='b')

    # Intrusion speed
    z_ax[3].scatter(100*float(data_array[i,8]), 100*float(data_array[i,7]), marker=markers[md['H']], color='b')

H05 = z_ax[0].scatter(100,100,marker=markers[0.05],color='b')
H10 = z_ax[0].scatter(100,100,marker=markers[0.10],color='b')
H15 = z_ax[0].scatter(100,100,marker=markers[0.15],color='b')
H20 = z_ax[0].scatter(100,100,marker=markers[0.2],color='b')

H05r = z_ax[0].scatter(100,100,marker=markers[0.05],color='r')
H10r = z_ax[0].scatter(100,100,marker=markers[0.10],color='r')
H15r = z_ax[0].scatter(100,100,marker=markers[0.15],color='r')
H20r = z_ax[0].scatter(100,100,marker=markers[0.2],color='r')

z_ax[0].set_xlabel("$z_{max}$ (theory) (cm)")
z_ax[0].set_ylabel("$z_{max}$ (simulation) (cm)")

z_ax[1].set_xlabel("$z_{n}$ (theory) (cm)")
z_ax[1].set_ylabel("$z_{n}$ (simulation) (cm)")

z_ax[2].set_ylabel("$z_{ss}/z_{max}$")
z_ax[2].set_xlabel("$Fr_i = w_i/(b_i r_i)^{1/2}$")

z_ax[3].set_xlabel("$M_n/Q_n \,(cm \cdot s^{{-1}})$")
z_ax[3].set_ylabel("$V_r\, (cm \cdot s^{{-1}})$")

z_ax[0].set_xlim(4, 10.5)
z_ax[0].set_ylim(4, 10.5)
z_ax[1].set_xlim(0, 4)
z_ax[1].set_ylim(0, 4)

z_ax[2].set_xlim(1, 4.5)
z_ax[2].set_ylim(0.4, 1.2)

z_ax[3].set_xlim(2, 6)
z_ax[3].set_ylim(0.2, 1.5)

z_ax[0].set_title("(a)")
z_ax[1].set_title("(b)")
z_ax[2].set_title("(c)")
z_ax[3].set_title("(d)")

#ax[0].legend((H05, H10, H15, H20), ("H = 5 cm", "H = 10 cm", "H = 15 cm", "H = 20 cm"),
        #facecolor=(0.9,0.9,0.9), loc='lower right', fontsize=10)
z_ax[0].legend((H10, H15, H20), ("H = 10 cm", "H = 15 cm", "H = 20 cm"),
        loc='lower right', fontsize=10)

z_ax[0].set_aspect(1)
z_ax[1].set_aspect(1)

z_ax[2].set_aspect(3.5/0.8)
z_ax[3].set_aspect(4/1.3)

z_fig.savefig('/home/cwp29/Documents/4report/figs/zcomp.png',dpi=200)
#v_fig.savefig('/home/cwp29/Documents/4report/figs/vcomp.png',dpi=200)
plt.show()
