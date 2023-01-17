import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from os.path import join
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_plotindex, get_index, compute_F0
from scipy import ndimage, interpolate, spatial

#TODO modify to display SAME simulation at DIFFERENT times

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
convex_hull = True

dirs = ["N_1", "N_1", "N_1"]
start_times = [4, 4, 4]
plot_times = [4.25, 10, 14]

##### ---------------------- #####

#d Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Parent directory: ", save_dir)

# Get simulation metadata
md = get_metadata(join(save_dir, dirs[0]), version)

gxf, gyf, gzf, dzf = get_grid(join(save_dir, dirs[0], 'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(save_dir, dirs[0], 'grid.h5'), md, fractional_grid=False)

# Get data
bs = []
ts = []
mds = []
Fs = []
scatters = []
scatter_fluxs = []
bmins = []
bmaxs = []
tmins = []
tmaxs = []
bbins = []
tbins = []
timess = []
b_zmaxs = []
t_rmss = []
Xs = []
Ys = []
Xfs = []
Yfs = []

NSAMP = -1
for d in range(len(dirs)):
    md = get_metadata(join(save_dir, dirs[d]), version)
    mds.append(md)
    print("Metadata: ", md)

    with h5py.File(join(save_dir,dirs[d],"movie.h5"), 'r') as f:
        print("Keys: %s" % f.keys())
        time_keys = list(f['th1_xz'])
        print(time_keys)
        # Get buoyancy data
        t = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
        b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
        t = g2gf_1d(md, t)
        b = g2gf_1d(md, b)
        scatter = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
        scatter_flux = np.array([np.array(f['td_flux'][t]) for t in time_keys])
        times = np.array([f['th1_xz'][t].attrs['Time'] for t in time_keys])
        print(times)

        if NSAMP > 0:
            if len(times) != NSAMP:
                print("ERROR: mismatched simulation lengths")
        else:
            NSAMP = len(times)

        f.close()

    scatter = np.where(scatter == 0, np.nan, scatter)

    plot_max = 1.6*mds[d]['H']
    plot_min = 0.95*mds[d]['H']
    start_idx = get_index(start_times[d], times)
    print("Starting at t = {0:.2f} s".format(times[start_idx]))

    Fs.append(compute_F0(join(save_dir, dirs[d]), md, tstart_ind = start_idx,
        verbose=False, tracer=False))

    idx_minf = get_plotindex(plot_min, gzf)-1
    idx_maxf = get_plotindex(plot_max, gzf)

    idx_min = idx_minf
    idx_max = idx_maxf+1

    print(plot_min, plot_max)
    print(gz[idx_min:idx_max], len(gz[idx_min:idx_max]))
    print(gzf[idx_minf:idx_maxf], len(gzf[idx_minf:idx_maxf]))

    X, Y = np.meshgrid(gx, gz[idx_min:idx_max])
    Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf])

    bref = b[0, :, int(md['Nx']/2)]
    t_orig = t
    b = b[start_idx:, idx_minf:idx_maxf, :]
    t = t[start_idx:, idx_minf:idx_maxf, :]
    scatter = scatter[start_idx:]
    scatter_flux = scatter_flux[start_idx:]
    times = times[start_idx:]

    bmin = 0
    bmax = md['b_factor']*md['N2']*(md['LZ']-md['H'])

    F0 = md['b0']*(md['r0']**2)
    alpha = md['alpha_e']

    tmin = 5e-4
    tmax = md['phi_factor']*5*F0 / (3 * alpha) * np.power(0.9*alpha*F0, -1/3) * np.power(
            md['H']+ 5*md['r0']/(6*alpha), -5/3)

    Nb = int(md['Nb'])
    Nt = int(md['Nphi'])
    db = (bmax - bmin)/Nb
    dt = (tmax - tmin)/Nt
    dx = md['LX']/md['Nx']
    dy = md['LY']/md['Ny']
    bbin = [bmin + (i+0.5)*db for i in range(Nb)]
    tbin = [tmin + (i+0.5)*dt for i in range(Nt)]

    print(bmin,bmax)
    print(tmin,tmax)

    # Compute z_max from simulations

    tracer_data_vert = t_orig[1:, :, int(md['Nx']/2)]
    tracer_thresh = 5e-4
    plume_vert = np.where(tracer_data_vert > tracer_thresh, 1, 0)

    heights = []
    for i in range(len(plume_vert)):
        stuff = np.where(plume_vert[i] == 1)[0]
        if len(stuff) == 0:
            heights.append(0)
        else:
            heights.append(gzf[np.max(stuff)])

    zmax_exp = np.max(heights)

    f = interpolate.interp1d(gzf, bref)
    b_zmax = f(zmax_exp)

    # Compute mean tracer value entering computation domain
    tracer_data = t[:, 0, int(md['Nx']/2)]

    t_rms = np.sqrt(np.mean(np.power(tracer_data,2)))

    bmins.append(bmin)
    bmaxs.append(bmax)
    tmins.append(tmin)
    tmaxs.append(tmax)
    bbins.append(bbin)
    tbins.append(tbin)

    Xs.append(X)
    Ys.append(Y)
    Xfs.append(Xf)
    Yfs.append(Yf)

    b_zmaxs.append(b_zmax)
    t_rmss.append(t_rms)

    timess.append(times)
    bs.append(b)
    ts.append(t)
    scatters.append(scatter)
    scatter_fluxs.append(scatter_flux)

#############################################################################################################

bs = np.array(bs)
ts = np.array(ts)
Fs = np.array(Fs)
scatters = np.array(scatters)
mds = np.array(mds)
bmins = np.array(bmins)
bmaxs = np.array(bmaxs)
tmins = np.array(tmins)
tmaxs = np.array(tmaxs)
b_zmaxs = np.array(b_zmaxs)
t_rmss = np.array(t_rmss)
Xs = np.array(Xs)
Xfs = np.array(Xfs)
Ys = np.array(Ys)
Yfs = np.array(Yfs)
timess = np.array(timess)

#############################################################################################################
# Figure set-up

fig, ax = plt.subplots(2, len(dirs), figsize=(6*len(dirs),6), constrained_layout=True)

for d in range(len(dirs)):
    scatter = scatters[d]
    scatter_flux = scatter_fluxs[d]
    b = bs[d]
    t = ts[d]
    md = mds[d]
    bbin = bbins[d]
    tbin = tbins[d]
    tmax = tmaxs[d]
    tmin = tmins[d]
    bmax = bmaxs[d]
    bmin = bmins[d]
    b_zmax = b_zmaxs[d]
    t_rms = t_rmss[d]
    X = Xs[d]
    Y = Ys[d]
    Xf = Xfs[d]
    Yf = Yfs[d]
    times = timess[d]

    Nb = int(md['Nb'])
    Nt = int(md['Nphi'])
    db = (bmax - bmin)/Nb
    dt = (tmax - tmin)/Nt

    plot_idx = get_index(plot_times[d], times)

    #########################################################

    sx, sy = np.meshgrid(bbin, tbin)

    #########################################################

    contours_b = np.linspace(0, np.max(b), 10)

    #########################################################

    im_scatter = ax[1,d].pcolormesh(sx, sy, scatter[plot_idx], cmap='plasma')
    im_scatter.set_clim(0, np.nanmax(scatter[-1]))

    trac_im = ax[0,d].pcolormesh(X, Y, t[plot_idx], cmap='viridis')
    trac_im.set_clim(0, np.nanmax(t[-1]))
    b_cont = ax[0,d].contour(Xf, Yf, np.where(t[plot_idx] <= 5e-4, b[plot_idx], np.NaN),
                levels = contours_b, cmap='cool', alpha=0.8)
    b_cont_fill = ax[0,d].contourf(b_cont, levels = contours_b, cmap='cool', alpha=0.8, extend='min')
    t_cont = ax[0,d].contour(Xf, Yf, t[plot_idx], levels = [5e-4], colors='k', alpha=0.8)

    #########################################################

    sx_nan = sx[~np.isnan(scatter[plot_idx]+scatter_flux[plot_idx+1])].flatten()
    sy_nan = sy[~np.isnan(scatter[plot_idx]+scatter_flux[plot_idx+1])].flatten()
    points = np.array(list(zip(sx_nan,sy_nan)))

    if len(points) > 0:
        hull = spatial.ConvexHull(points)

        ax[1,d].plot(points[hull.simplices[0],0], points[hull.simplices[0],1], 'r--', label="convex hull")
        for simplex in hull.simplices:
            ax[1,d].plot(points[simplex,0], points[simplex,1], 'r--')

    #########################################################

    if d == len(dirs)-1:
        # Decorations
        #im_div = make_axes_locatable(ax[0,d])
        #im_cax = im_div.append_axes("right", size="5%", pad=0.05)
        im_cb = plt.colorbar(trac_im, ax = ax[0,d], label="tracer concentration")

        cont_norm = matplotlib.colors.Normalize(vmin=b_cont.cvalues.min(), vmax=b_cont.cvalues.max())
        cont_sm = plt.cm.ScalarMappable(norm=cont_norm, cmap=b_cont.cmap)
        #im_cont_cax = im_div.append_axes("right", size="5%", pad=0.75)
        im_cont_cb = plt.colorbar(b_cont_fill, ax = ax[0,d], format=lambda x,_ :f"{x:.3f}",
                label=r"buoyancy ($m\, s^{-2}$)")
        im_cont_cb.set_alpha(1)
        im_cont_cb.draw_all()

        corr_div = make_axes_locatable(ax[1,d])
        #corr_cax = corr_div.append_axes("right", size="5%", pad=0.05)
        corr_cb = plt.colorbar(im_scatter, ax = ax[1,d], label=r"volume ($m^3$)")

    labels = ["(a)", "(b)", "(c)"]
    ax[0,d].set_title("{2} $N^2 = ${0:.2f}, t = {1:.2f} s".format(md['N2'], times[plot_idx], labels[d]))

    ax[1,d].set_xlabel(r"buoyancy ($m\,s^{-2}$)")
    if d == 0:
        ax[1,d].set_ylabel("tracer conc.")
        ax[0,d].set_ylabel("z (m)")
    ax[0,d].set_xlabel("x (m)")
    ax[1,d].set_xlim(bmin, bmax)
    ax[1,d].set_ylim(tmin, tmax)

    #ax[1,d].axvline(b_zmax, color='k', linestyle='--')
    #ax[1,d].axhline(t_rms, color='k', linestyle='--')
    ax[1,d].legend()

    ax[0,d].set_xlim(0.2, 0.4)

#plt.tight_layout()
plt.savefig('/home/cwp29/Documents/essay/figs/comp_pdf.png', dpi=300)
plt.show()
