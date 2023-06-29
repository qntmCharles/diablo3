import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py, gc, sys
import numpy as np
from scipy.interpolate import griddata
from scipy import integrate
import matplotlib
from datetime import datetime
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join, isfile
from os import listdir
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_index, get_plotindex, compute_F0

def compute_pdf(data, ref, bins, normalised=False):
    out_bins = [0 for i in range(len(bins)-1)]

    for i in range(len(bins)-1):
        out_bins[i] = np.sum(np.where(np.logical_and(data >= bins[i],
            data < bins[i+1]), ref, 0))

    out_bins = np.array(out_bins)

    if normalised:
        area = integrate.trapezoid(np.abs(out_bins), 0.5*(bins[1:]+bins[:-1]))
        return out_bins/area
    else:
        return out_bins

##### USER-DEFINED PARAMETERS #####

params_file = "params.dat"
save = False

##### ----------------------- #####

##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(run_dir, version)
print("Complete metadata: ",md)

##### Get grid #####
gxf, gyf, gzf, dzf = get_grid(join(run_dir, 'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(run_dir, 'grid.h5'), md, fractional_grid=False)

##### Get data #####

with h5py.File(join(save_dir,"mean.h5"), 'r') as f:
    print("Mean keys: %s" % f.keys())
    time_keys = list(f['tb_source'])
    print(time_keys)
    bbins_file = np.array(f['tb_strat_bins']['0001'])
    source_dists = np.array([np.array(f['tb_source'][t]) for t in time_keys])
    strat_dists = np.array([np.array(f['tb_strat'][t]) for t in time_keys])
    times = np.array([float(f['tb_source'][t].attrs['Time']) for t in time_keys])
    t_me = np.array([np.array(f['thme02'][t]) for t in time_keys])*md['Ny']*md['Nx']
    NSAMP = len(times)

    f.close()

with h5py.File(join(save_dir,"movie.h5"), 'r') as f:
    b = np.array(f['th1_xz']['0005'])

centreline_b = b[:,int(md['Nx']/2)]

bin_end = int(np.where(bbins_file == -1)[0][0])
bbins_file = bbins_file[:bin_end]
strat_dists = strat_dists[:,:bin_end]

print(NSAMP)

plot_max = 1.45*md['H']
plot_min = 0.99*md['H']

idx_minf = get_plotindex(plot_min, gzf)-1
idx_maxf = get_plotindex(plot_max, gzf)

idx_min = idx_minf
idx_max = idx_maxf+1

##### Non-dimensionalisation #####

F0 = compute_F0(save_dir, md, tstart_ind = 6*4, verbose=False, zbot=0.7, ztop=0.95, plot=False)
N = np.sqrt(md['N2'])

T = np.power(N, -1)
L = np.power(F0, 1/4) * np.power(N, -3/4)
Bdim = L * np.power(T, -2)

gz -= md['H']
gz /= L
gzf -= md['H']
gzf /= L

bbins_file /= Bdim

times /= T

print(F0)
print(md['b0']*md['r0']*md['r0'])

##### ====================== #####

############################################################################################################
# Tracer vs. buoyancy heatmap
############################################################################################################

bt_pdfs = strat_dists[:,:-1]
print(bt_pdfs)
zt_pdfs = t_me[:, idx_minf:idx_maxf+1]

bt_pdfs = np.swapaxes(bt_pdfs, axis1=1, axis2=0)
zt_pdfs = np.swapaxes(zt_pdfs, axis1=1, axis2=0)

Tz, Z = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), gz[idx_min:idx_max+1])
Tzf, Zf = np.meshgrid(times+md['SAVE_STATS_DT']/2, gzf[idx_min:idx_max])
Tb, B = np.meshgrid(np.append(times, times[-1]+md['SAVE_STATS_DT']), bbins_file)
Tbf, Bf = np.meshgrid(times+md['SAVE_STATS_DT']/2, 0.5*(bbins_file[1:]+bbins_file[:-1]))

print(bt_pdfs.shape)
print(bbins_file.shape)
print(strat_dists.shape)
print(times.shape)
print(Tb.shape)

fig, ax = plt.subplots(1,2, figsize=(12, 3.5), constrained_layout=True)

bt_im = ax[0].pcolormesh(Tb, B, bt_pdfs, shading='flat', cmap='viridis')

zt_im = ax[1].pcolormesh(Tz, Z, zt_pdfs, shading='flat', cmap='viridis')


print(centreline_b[idx_min-5:idx_min+5])
print(idx_min)
print(centreline_b[idx_min])
ax[0].set_ylim(centreline_b[idx_min]/Bdim, centreline_b[idx_max]/Bdim)
ax[1].set_ylim((plot_min-md['H'])/L, (plot_max-md['H'])/L)

bt_cont = ax[0].contour(Tbf, Bf, bt_pdfs, levels=[1e1], colors='r', alpha=0.7)
bt_cbar = plt.colorbar(bt_im, ax=ax[0], label="tracer conc.")
bt_cbar.add_lines(bt_cont)

#zt_cont = ax[1].contour(Tzf, Zf, zt_pdfs, levels=[1e1], colors='r', alpha=0.7)
#zt_cont = ax[1].contour(Tzf, Zf, zt_pdfs, levels=[50], colors='r', alpha=0.7, linestyle='--')
zt_cbar = plt.colorbar(zt_im, ax=ax[1], label="tracer conc.")
#zt_cbar.add_lines(zt_cont)

t_step = int(round(2.0 / md['SAVE_STATS_DT'], 0))
print(t_step)
for i in range(0, NSAMP, t_step):
    print(times[i])
    ax[0].axvline(times[i], linestyle='--', color='gray', alpha=0.5)
    ax[1].axvline(times[i], linestyle='--', color='gray', alpha=0.5)

#ax[0].set_title("tracer vs. buoyancy heatmap")
ax[0].set_xlabel("t")
ax[0].set_ylabel("buoyancy")
#ax[1].set_title("tracer vs. height heatmap")
ax[1].set_xlabel("t")
ax[1].set_ylabel("z")

ax[0].set_xlim(4/T, 15/T)
ax[1].set_xlim(4/T, 15/T)

#fig.savefig('/home/cwp29/Documents/papers/draft/figs/hmap.png', dpi=300)
#fig.savefig('/home/cwp29/Documents/papers/draft/figs/hmap.pdf', dpi=300)
plt.show()
