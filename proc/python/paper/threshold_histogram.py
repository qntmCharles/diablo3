import sys, os
import matplotlib.colors as colors
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_plotindex, get_index, compute_F0
from scipy import ndimage
from skimage import filters, exposure
import numpy.polynomial.polynomial as poly
from scipy.signal import argrelextrema

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

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
idx_maxf = get_plotindex(plot_max, gzf)

idx_min = idx_minf
idx_max = idx_maxf+1

print(idx_min, idx_max)

gx, gy, dz = np.meshgrid(gxf, dz[idx_min:idx_max], gyf, indexing='ij', sparse=True)
volume = (md['LX']/md['Nx'])**2 * dz
print(volume.shape)

print("Complete metadata: ", md)

with h5py.File(join(save_dir, "movie.h5"), 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)

    svd = np.array([f['pvd'][t] for t in time_keys])
    vd = np.array([f['td_scatter'][t] for t in time_keys])
    S = np.array([np.array(f['td_flux'][t]) for t in time_keys])
    times = np.array([f['pvd'][t].attrs['Time'] for t in time_keys])
    NSAMP = len(times)

M = np.copy(svd)
for i in range(1, NSAMP):
    S[i] += S[i-1]
    M[i] *= np.sum(S[i])


with h5py.File(save_dir+"/mean.h5", 'r') as f:
    bbins = np.array(f['PVD_bbins']['0001'])
    phibins = np.array(f['PVD_phibins']['0001'])

db = bbins[1] - bbins[0]
dphi = phibins[1] - phibins[0]

##### Non-dimensionalisation #####

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
B = L * np.power(T, -2)
V = L * L * L

times /= T
db /= B
M /= V
vd /= V

##### ====================== #####
sim_step = 58

svd = np.where(svd == 0, np.nan, svd)

svd_lim = np.nanmax(svd[sim_step])

#svd_bins = np.linspace(np.nanmin(svd[30:]), np.nanmax(svd[30:]), 100)
svd_bins = np.linspace(0, svd_lim, 100)
bins_plot = 0.5 * (svd_bins[1:] + svd_bins[:-1])


results = np.zeros(shape=(NSAMP, len(svd_bins)))
results2 = np.zeros(shape=(NSAMP, len(svd_bins)))

for i in range(1,NSAMP):
    for j in range(1, len(svd_bins)):
        results[i, j] = np.nansum(
            np.where(np.logical_and(svd[i] <= svd_bins[j], svd[i] > 0), vd[i], 0))
        results2[i, j] = np.nansum(
                np.where(svd[i] > svd_bins[j], vd[i], 0))

dOmegaInt_dt = np.gradient(results, times, axis=0)
comp_dOmegaInt_dt = np.gradient(results2, times, axis=0)


print("Done getting data.")

fig = plt.figure(figsize=(8, 2.5))
ax = plt.gca()

N = 6
col = plt.cm.viridis(np.linspace(0, 1, N))

mean_results = ndimage.uniform_filter1d(results, axis=0, size=5, mode='nearest')
mean_dOmegaInt_dt = np.gradient(mean_results, times, axis=0)
for step, c in zip(range(-N//2, N//2 + 1), col):
    if step != 0:
        ax.plot(svd_bins, mean_dOmegaInt_dt[sim_step+step], color=c, alpha=0.5,
                label=r"$t = {0:.2f}$".format(times[sim_step+step]))
    else:
        ax.plot(svd_bins, mean_dOmegaInt_dt[sim_step], color='b',
                label=r"$t = {0:.2f}$".format(times[sim_step+step]))

#mean_dOmegaInt_dt = np.mean(dOmegaInt_dt[-N:], axis=0)
#mean_dOmegaInt_dt = np.gradient(np.mean(results[-N:], axis=0), times, axis=0)
#for step, c in zip(range(-N, 0), col):
    #ax.plot(svd_bins, dOmegaInt_dt[step], label=r"$t = {0:.2f}$".format(times[step]), color=c, alpha=0.5)
    #ax.plot(svd_bins, mean_dOmegaInt_dt[step], label=r"$t = {0:.2f}$".format(times[step]), color=c, alpha=0.5)

#ax.plot(svd_bins, mean_dOmegaInt_dt, label="mean", color='b')

#ax.set_ylim(-0.1, 0.1)
ax.set_xlim(0, svd_lim)

ax.legend(loc='center right', fancybox=True, shadow=True)

ax.axhline(0, color='k')

#ax.set_ylabel(r"$\mathcal{E}$")
ax.set_xlabel(r"$\hat{\Omega}^*$")
ax.set_ylabel(r"$\frac{\mathrm{d}I_{\hat{\Omega}^*}}{\mathrm{d}t}$", rotation=0)

plt.tight_layout()
#plt.savefig('/home/cwp29/Documents/papers/draft/figs/threshold.png', dpi=300)
#plt.savefig('/home/cwp29/Documents/papers/draft/figs/threshold.pdf')


vd = np.where(vd == 0, np.nan, vd)
min_vol = np.power(md['LX']/md['Nx'], 2) * md['LZ']/md['Nz']
min_vol /= V
vd = np.where(vd > 50*min_vol * (64/md['Nb'])**2, vd, np.NaN)
M = np.where(np.isnan(vd), np.nan, M)
Mpos = np.where(M > 0, M, np.nan)

db = bbins[1] - bbins[0]
dphi = phibins[1] - phibins[0]

sx, sy = np.meshgrid(np.append(bbins-db/2, bbins[-1]+db/2),
        np.append(phibins - dphi/2, phibins[-1] + dphi/2))

print(M[i].shape)

idx_lim = 12

for i in range(20, NSAMP):
    fig, ax = plt.subplots(1, 2)
    im = Mpos[i][np.isfinite(Mpos[i])]
    if len(im) == 0:
        print("fuck")
        continue
    m = filters.threshold_otsu(im)
    hist, bins_center = exposure.histogram(im, nbins=32)

    c = poly.polyfit(bins_center[:idx_lim], hist[:idx_lim], 3)
    fit = poly.polyval(bins_center[:idx_lim], c)
    a = argrelextrema(fit, np.less)

    ax[0].plot(bins_center, hist, lw=2)
    ax[0].axvline(m, color='k', ls='--')
    ax[0].plot(bins_center[:idx_lim], fit, color='g')
    ax[0].set_title(times[i])

    if len(a[0]) > 0:
        thresh = bins_center[a[0][0]]
        ax[0].axvline(thresh, color='g', ls='--')
    else:
        print("Threshold undefined")
        continue


    nodes = [-np.nanmax(M[i]), 0, 0, thresh, thresh, np.nanmax(M[i])]
    custom_colors = ["blue", plt.cm.Blues(0.5), plt.cm.Greens(0.5), "green", plt.cm.Reds(0.5),
            "red"]
    norm = plt.Normalize(min(nodes), max(nodes))
    custom_cmap = colors.LinearSegmentedColormap.from_list("", list(zip(map(norm,nodes), custom_colors)))

    im_M_bphi = ax[1].pcolormesh(sx, sy, M[i], cmap=custom_cmap, norm=norm)

    plt.show()
