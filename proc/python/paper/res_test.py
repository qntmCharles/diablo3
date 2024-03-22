import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors
from mpl_toolkits import axes_grid1
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, compute_F0, get_plotindex, get_index
from os.path import join, isfile
from os import listdir

from scipy.interpolate import griddata, interp1d
from scipy import integrate, ndimage, spatial

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

save = True
show = not save

fig_save_dir = '/home/cwp29/Documents/papers/conv_pen/draft3/figs'

b_thresh = 1e-4

bmax_plot = 4
phimax_plot = 0.15
factor = 1.0
aspect = factor*bmax_plot/phimax_plot

tplot = 10

clim_flag = True

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data acquisition
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get dir locations from param file
base_dir = '/home/cwp29/diablo3/strat/res_test/'
version = 3.8
save_dirs = ['final/128', 'final/256', 'final/512_4', 'final/1024']

fig6, axs = plt.subplots(1,len(save_dirs), figsize=(8, 2), constrained_layout=True, sharey=True)

for d in range(len(save_dirs)):
    save_dir = join(base_dir, save_dirs[d])

    # Get simulation metadata
    md = get_metadata(save_dir, version)
    md['kappa'] = md['nu']/0.7

    print("Complete metadata: ", md)

    # Get data
    with h5py.File(join(save_dir, 'movie.h5'), 'r') as f:
        print("Keys: %s" % f.keys())
        time_keys = list(f['th1_xz'])
        # Get buoyancy data
        th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
        W = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
        NSAMP = len(W)
        times = np.array([float(f['th1_xz'][t].attrs['Time']) for t in time_keys])

        f.close()

    with h5py.File(join(save_dir,"mean.h5"), 'r') as f:
        print("Mean keys: %s" % f.keys())
        time_keys = list(f['tb_source'])
        bbins = np.array(f['PVD_bbins']['0001'])
        phibins = np.array(f['PVD_phibins']['0001'])

        mean_times = np.array([float(f['tb_source'][t].attrs['Time']) for t in time_keys])
        assert np.array_equal(times, mean_times)
        mean_NSAMP = len(times)
        assert mean_NSAMP == NSAMP

        f.close()

    with open(join(save_dir, "time.dat"), 'r') as f:
        reset_time = float(f.read())
        print("Plume penetration occured at t={0:.4f}".format(reset_time))

        if len(np.argwhere(times == 0)) > 1:
            t0_idx = np.argwhere(times == 0)[1][0]
            t0 = times[t0_idx-1]

            for i in range(t0_idx):
                times[i] -= reset_time

    centreline_phi = np.mean(np.mean(th2_xz[:,:,int(md['Nx']/2)-1:int(md['Nx']/2)+2], axis=2), axis=0)
    centreline_phi = np.mean(th2_xz[:,:,int(md['Nx']/2)], axis=0)
    phi0 = np.max(centreline_phi)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Non-dimensionalisation
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
    N = np.sqrt(md['N2'])

    T = np.power(N, -1)
    L = np.power(F0, 1/4) * np.power(N, -3/4)
    B = L * np.power(T, -2)
    U = L / T
    V = L * L * L

    times /= T
    bbins /= B
    phibins /= phi0
    db = bbins[1] - bbins[0]
    dphi = phibins[1] - phibins[0]

    md['SAVE_STATS_DT'] /= T

    W /= (V*db*B*dphi*phi0)

    sx, sy = np.meshgrid(np.append(bbins-db/2, bbins[-1]+db/2),
            np.append(phibins - dphi/2, phibins[-1] + dphi/2))

    W = np.where(W == 0, np.nan, W)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Appendix A figure 2: volume distribution W and source S evolution
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Need: W_thresh, sx, sy, bbins, phibins, db, dphi

    im_W = axs[d].pcolormesh(sx, sy, W[get_index(tplot, times)]/np.nansum(W[get_index(tplot, times)]),
            cmap='plasma')
    axs[d].set_aspect(aspect)

    #im_W.set_clim(0, 0.2)

    if d == 0:
        axs[d].set_ylabel(r"$\phi$", rotation=0, labelpad=10)
        cb_W = fig6.colorbar(im_W, ax = axs[-1], label=r"$W$", location='right', extend='max')
        cb_W.formatter.set_powerlimits((0, 0))
    axs[d].set_xlabel(r"$b$", rotation=0, labelpad=10)


    if clim_flag:
        Wmax = .05*np.nanmax(W[get_index(tplot, times)]/np.nansum(W[get_index(tplot, times)]))
        clim_flag = False

    im_W.set_clim(Wmax)

    axs[d].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

    axs[d].set_xlim(bbins[0]-db/2, bmax_plot)
    axs[d].set_ylim(phibins[0]-dphi/2, phimax_plot)

for ax, label in zip(axs.ravel(), ['(a)', '(b)', '(c)', '(d)']):
    ax.text(-0.1, 1.15, label, transform=ax.transAxes, va='top', ha='right')


if save:
    fig6.savefig(join(fig_save_dir, 'res_test.png'), dpi=300)
    fig6.savefig(join(fig_save_dir, 'res_test.pdf'))

plt.show()
