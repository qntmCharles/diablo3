import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_plotindex, get_index
from scipy import ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
normalised=True

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(join(save_dir,'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(save_dir,'grid.h5'), md, fractional_grid=False)

plot_min = 0.95*md['H']
plot_max = md['LZ']

idx_minf = get_plotindex(plot_min, gzf)-1
idx_maxf = get_plotindex(plot_max, gzf)+1

idx_min = idx_minf
idx_max = idx_maxf+1

print(idx_min, idx_max)
X, Y = np.meshgrid(gx, gz[idx_min:idx_max])
Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf])

gx, gy, dz = np.meshgrid(gxf, dz[idx_min:idx_max], gyf, indexing='ij')
volume = (md['LX']/md['Nx'])**2 * dz
print(volume.shape)

bmin = 0
bmax = md['b_factor']*md['N2']*(md['LZ']-md['H'])

F0 = md['b0']*(md['r0']**2)
alpha = md['alpha_e']

phimin = 5e-4
phimax = md['phi_factor']*5*F0 / (3 * alpha) * np.power(0.9*alpha*F0, -1/3) * np.power(
        md['H']+ 5*md['r0']/(6*alpha), -5/3)

db = (bmax - bmin)/md['Nb']
dphi = (phimax - phimin)/md['Nphi']
bbins = np.array([bmin + (i+0.5)*db for i in range(int(md['Nb']))])
phibins = np.array([phimin + (i+0.5)*dphi for i in range(int(md['Nphi']))])

print("Complete metadata: ", md)

with h5py.File(join(save_dir, "movie.h5"), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)

    bt_vd = np.array([f['td_scatter'][t] for t in time_keys])
    svd = np.array([f['svd'][t] for t in time_keys])
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    b = b[:, idx_min:idx_max, :]
    phi = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    phi = phi[:, idx_min:idx_max, :]
    print(phi.shape)

    step = len(b)-1

    f.close()

sx, sy = np.meshgrid(bbins, phibins)

s_cvals = [-0.7*np.nanmax(svd[-1]), 0, 0.7*np.nanmax(svd[-1])]
s_norm=plt.Normalize(min(s_cvals),max(s_cvals))

test_array = np.zeros_like(b[step])
for k in range(int(md['Nb'])):
    for j in range(int(md['Nphi'])):
        test_array[np.logical_and(np.logical_and(b[step]> bbins[k] - db/2,
            b[step] <= bbins[k] + db/2),np.logical_and(phi[step] > phibins[j] - dphi/2,
            phi[step] <= phibins[j] + dphi/2))] =  svd[step, j, k]

test_array = np.where(test_array == 0, np.nan, test_array)
svd = np.where(svd == 0, np.nan, svd)

pdf_fig = plt.figure(constrained_layout=True, figsize=(12, 8))
ax = pdf_fig.subplot_mosaic("ABC;DEF",
        gridspec_kw={'wspace':0.1, 'hspace':0.1})

summary_fig = plt.figure(constrained_layout=True, figsize=(12, 8))
sum_ax = summary_fig.subplot_mosaic("GH;II")

hist_bins = np.concatenate((np.linspace(np.nanmin(svd[step]), -md['Omega_thresh'], 10),
    np.array([0, md['Omega_thresh']]),
    np.linspace(md['Omega_thresh'], np.nanmax(svd[step]), 10)[1:]))
N, bins, patches = sum_ax["I"].hist(svd.flatten(), bins = hist_bins,
        weights = bt_vd.flatten(), density=True)
for b, p in zip(bins, patches):
    if b < -md['Omega_thresh']:
        p.set_facecolor('b')
    elif b < md['Omega_thresh']:
        p.set_facecolor('g')
    else:
        p.set_facecolor('r')

cvals = [-1, 0, 1]
colors = ["b","g","r"]
norm=plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(map(norm,cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

svd[step] = np.where(svd[step] > md['Omega_thresh'], 1, svd[step])
svd[step] = np.where(svd[step] < -md['Omega_thresh'], -1, svd[step])
svd[step] = np.where(np.logical_and(svd[step] >= -md['Omega_thresh'], svd[step] <= md['Omega_thresh']), 0,
        svd[step])
sum_ax["G"].pcolormesh(sx, sy, svd[step], cmap=cmap)
sum_ax["G"].set_aspect(bmax/phimax)

test_array = np.where(test_array > md['Omega_thresh'], 1, test_array)
test_array = np.where(test_array < -md['Omega_thresh'], -1, test_array)
test_array = np.where(np.logical_and(test_array >= -md['Omega_thresh'], test_array <= md['Omega_thresh']), 0,
        test_array)

#ax[1].pcolormesh(X, Y, test_array, cmap='coolwarm', norm=s_norm)
sum_ax["H"].pcolormesh(X, Y, test_array, cmap=cmap)
phi_cont = sum_ax["H"].contour(Xf, Yf, phi[step], levels = [5e-4], colors='k', alpha=0.8)
sum_ax["H"].set_xlim(0.2, 0.4)
sum_ax["H"].set_ylim(0.95*md['H'], 1.5*md['H'])
sum_ax["H"].set_aspect(1)

fields = ["Ri", "chi", "Re_b", "tke", "e", "B"]
labels = [r"$\mathrm{Ri}$", r"$\chi$", r"$\log \mathrm{Re}_b$", r"$\log \varepsilon$", r"Diapycnal flux $e$",
        r"$B$"]

for i, key, l in zip(range(len(fields)), ["A", "B", "C", "D", "E", "F"], labels):
    with h5py.File(join(save_dir, "mean.h5"), 'r') as f:
        print("Keys: %s" % f.keys())
        time_keys = list(f['urms'])
        print(time_keys)

        bbins = np.array(f['SVD_bbins'][time_keys[0]])
        phibins = np.array(f['SVD_phibins'][time_keys[0]])

        pdf_plume = np.array([np.array(f[fields[i]+'_pdf_plume'][t]) for t in time_keys])
        pdf_mixed = np.array([np.array(f[fields[i]+'_pdf_mixed'][t]) for t in time_keys])
        pdf_mixing = np.array([np.array(f[fields[i]+'_pdf_mixing'][t]) for t in time_keys])
        print(pdf_plume.shape)

        pdf_bins = np.array(f[fields[i]+'_pdf_bins'][time_keys[0]])

        pdf_mixed_w = np.array([float(f[fields[i]+'_pdf_mixed_w'][t][0]) for t in time_keys])
        pdf_mixing_w = np.array([float(f[fields[i]+'_pdf_mixing_w'][t][0]) for t in time_keys])
        pdf_plume_w = np.array([float(f[fields[i]+'_pdf_plume_w'][t][0]) for t in time_keys])
        print(pdf_plume_w)

        total_weight = pdf_mixed_w + pdf_mixing_w + pdf_plume_w

        f.close()

    pdf_plot = (pdf_bins[1:] + pdf_bins[:-1])/2

    if normalised:
        ax[key].plot(pdf_plot, pdf_mixing[step,:-1], color='g',
                label="{0:.4f}".format(pdf_mixing_w[step]/total_weight[step]))
        ax[key].plot(pdf_plot, pdf_mixed[step,:-1], color='r',
                label="{0:.4f}".format(pdf_mixed_w[step]/total_weight[step]))
        ax[key].plot(pdf_plot, pdf_plume[step,:-1], color='b',
                label="{0:.4f}".format(pdf_plume_w[step]/total_weight[step]))
    else:
        ax[key].plot(pdf_plot, pdf_mixing_w[step]*pdf_mixing[step,:-1], color='g',
                label="{0:.4f}".format(pdf_mixing_w[step]/total_weight[step]))
        ax[key].plot(pdf_plot, pdf_mixed_w[step]*pdf_mixed[step,:-1], color='r',
                label="{0:.4f}".format(pdf_mixed_w[step]/total_weight[step]))
        ax[key].plot(pdf_plot, pdf_plume_w[step]*pdf_plume[step,:-1], color='b',
                label="{0:.4f}".format(pdf_plume_w[step]/total_weight[step]))

    if fields[i] == "Ri" or fields[i] == "e":
        ax[key].axvline(0, color='gray', linestyle='--', alpha=0.7)
    if fields[i] == "Ri":
        ax[key].axvline(0.25, color='gray', linestyle='--', alpha=0.7)

    ax[key].set_xlim(np.min(pdf_bins), np.max(pdf_bins))
    ax[key].set_title(l)
    ax[key].legend()

# Decorations
sum_ax["H"].set_xlabel("x (m)")
sum_ax["H"].set_ylabel("z (m)")
sum_ax["H"].set_title(r"$\hat{\Omega}}$ coloured spatial field")

sum_ax["G"].set_xlabel("buoyancy")
sum_ax["G"].set_ylabel("tracer conc.")
sum_ax["G"].set_title(r"Segregated volume distribution $\hat{\Omega}$")

mixing_line = mpatches.Patch(facecolor='g', edgecolor='g',
        label=r"$\left|\hat{{\Omega}}\right| < {0}$".format(md['Omega_thresh']))
mixed_line = mpatches.Patch(facecolor='r', edgecolor='r',
        label=r"$\hat{{\Omega}} > {0}$".format(md['Omega_thresh']))
plume_line = mpatches.Patch(facecolor='b', edgecolor='b',
        label=r"$\hat{{\Omega}} < -{0}$".format(md['Omega_thresh']))
sum_ax["G"].legend(handles=[mixing_line, mixed_line, plume_line])

sum_ax["I"].set_xlabel(r"$\hat{\Omega}$")
sum_ax["I"].set_ylabel("Volume (normalised)")
sum_ax["I"].set_title("Histogram of SVD values")

plt.show()
