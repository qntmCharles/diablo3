import numpy as np
from os.path import isfile
from matplotlib import pyplot as plt
from functions import get_metadata, get_grid, read_params, get_az_data

data_loc = '/home/cwp29/diablo3/strat/high_res/verif/as_data.npy'

if isfile(data_loc):
    data_array = np.load(data_loc, allow_pickle=False)
else:
    print("ERROR: Data file not found.")

fig, ax = plt.subplots(1,3, figsize=(15, 5))

markers = {
        0.2 : 'o',
        0.15 : '^',
        0.1 : 's',
        0.05 : '*'
        }

xs = np.linspace(0, 30, 5)
ys = np.linspace(0, 30, 5)
ax[0].plot(xs, ys, linestyle='--', color='k', alpha=0.5)
ax[1].plot(xs, ys, linestyle='--', color='k', alpha=0.5)
ax[2].plot(xs, 0.12*xs, linestyle='--', color='k', alpha=0.5)

for i in range(len(data_array)):
    md = get_metadata(data_array[i,0], "3.7")

    # Zmax
    ax[0].scatter(100*float(data_array[i,1]), 100*float(data_array[i,2]), marker=markers[md['H']], color='b')

    # Zeq
    ax[1].scatter(100*float(data_array[i,3]), 100*float(data_array[i,4]), marker=markers[md['H']], color='b')

    # Intrusion speed
    ax[2].scatter(100*float(data_array[i,6]), 100*float(data_array[i,5]), marker=markers[md['H']], color='b')

H05 = ax[0].scatter(100,100,marker=markers[0.05],color='b')
H10 = ax[0].scatter(100,100,marker=markers[0.10],color='b')
H15 = ax[0].scatter(100,100,marker=markers[0.15],color='b')
H20 = ax[0].scatter(100,100,marker=markers[0.2],color='b')

ax[0].set_xlabel("$z_{max}$ (theory) (cm)")
ax[0].set_ylabel("$z_{max}$ (experiment) (cm)")

ax[1].set_xlabel("$z_{n}$ (theory) (cm)")
ax[1].set_ylabel("$z_{n}$ (experiment) (cm)")

ax[2].set_xlabel("$M_n/Q_n \,(cm \cdot s^{{-1}})$")
ax[2].set_ylabel("$V_{experiment}\, (cm \cdot s^{{-1}})$")

ax[0].set_xlim(0, 30)
ax[0].set_ylim(0, 30)
ax[1].set_xlim(0, 30)
ax[1].set_ylim(0, 30)
ax[2].set_xlim(0, 12)
ax[2].set_ylim(0, 2)

ax[0].legend((H05, H10, H15, H20), ("H = 5 cm", "H = 10 cm", "H = 15 cm", "H = 20 cm"),
        loc='lower right', fontsize=10)

plt.tight_layout()
plt.show()
