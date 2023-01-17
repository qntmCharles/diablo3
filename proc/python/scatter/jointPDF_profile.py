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
from functions import get_metadata, read_params, get_grid, g2gf_1d, get_az_data
from scipy import ndimage, interpolate, spatial

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
convex_hull = True

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.floor(np.argmin(np.abs(griddata - z))))

##### ---------------------- #####

#d Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(join(save_dir,'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(save_dir,'grid.h5'), md, fractional_grid=False)

print("Complete metadata: ", md)

# Get data
with h5py.File(join(save_dir,"movie.h5"), 'r') as f:
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
    NSAMP = len(times)

plot_max = 1.6*md['H']
plot_min = 0.95*md['H']
start_idx = get_index(6, times)

idx_max = get_index(plot_max, gz)+1
idx_min = get_index(plot_min, gz)

idx_maxf = get_index(plot_max, gzf)
idx_minf = get_index(plot_min, gzf)

print(idx_min, idx_max)
print(idx_minf, idx_maxf)

X, Y = np.meshgrid(gx, gz[idx_min:idx_max])
Xf, Yf = np.meshgrid(gxf, gzf[idx_minf:idx_maxf])

bref = b[0, :, int(md['Nx']/2)]
t_orig = np.copy(t)
b_orig = np.copy(b)
b = b[start_idx:, idx_minf:idx_maxf, :]
t = t[start_idx:, idx_minf:idx_maxf, :]
scatter = scatter[start_idx:]
scatter_flux = scatter_flux[start_idx:]
times = times[start_idx:]
NSAMP = len(b)

bmin = 0
bmax = md['b_factor']*md['N2']*(md['LZ']-md['H'])
print(bmax)

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
bbins = [bmin + (i+0.5)*db for i in range(Nb)]
tbins = [tmin + (i+0.5)*dt for i in range(Nt)]

print(bmin,bmax)
print(tmin,tmax)

# accumulate fluxes
for i in range(1,NSAMP):
    scatter_flux[i] += scatter_flux[i-1]

scatter_corrected = scatter - scatter_flux

scatter_flux = np.where(scatter_flux == 0, np.nan, scatter_flux)

scatter_corrected = np.where(scatter_corrected == 0, np.nan, scatter_corrected)

scatter = np.where(scatter == 0, np.nan, scatter)

#########################################################
# Compute z_max from simulations

print(t_orig.shape)
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

t_centrelinemax = np.max(tracer_data)
t_bottommax = np.max(t[:,0,:])
t_totmax = np.max(t)
t_max = np.max(t[:,0,int(md['Nx']/2)])

#########################################################
# Figure set-up

while True:
    #fig,ax = plt.subplots(2, 2, figsize=(15,8))
    fig = plt.figure(constrained_layout = True, figsize=(15,6))
    ax = fig.subplot_mosaic(
        '''
        STJ
        ..B
        ''',
        gridspec_kw = {
            "width_ratios": [1, 0.25, 1],
            "height_ratios": [1, 0.3],
        }
    )

    fig.suptitle("time = {0:.2f} s".format(times[-1]))

    sx, sy = np.meshgrid(bbins, tbins)

    #########################################################
    # Create colour maps

    cvals = [-np.nanmax(scatter_flux[-1]), 0, np.nanmax(scatter_flux[-1])]
    colors = ["blue","white","red"]

    norm=plt.Normalize(min(cvals),max(cvals))
    tuples = list(zip(map(norm,cvals), colors))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

    s_cvals = np.concatenate((np.array([-0.5*np.nanmax(scatter_corrected[-1])]),
        np.linspace(0,np.nanmax(scatter_corrected[-1]),31)[:]))
    s_colors = np.concatenate((np.array([[0.,0.,1.,1.,]]),
        plt.cm.hot_r(np.linspace(0,1,31))))

    #s_cvals = np.concatenate((np.array([-1e-5, 0]), np.linspace(0,2e-5,31)[1:]))
    #s_colors = np.concatenate((np.array([[0.,0.,1.,1.,],[1.,1.,1.,1.]]), plt.cm.hot_r(np.linspace(0,1,30))))

    s_norm=plt.Normalize(min(s_cvals),max(s_cvals))
    s_tuples = list(zip(map(s_norm,s_cvals), s_colors))
    s_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", s_tuples, N=60)

    contours_b = np.linspace(0, np.max(b), 10)

    #########################################################
    # Compute tracer and buoyancy profiles

    p_idx = get_index(0.95*md['H'], gzf)

    t_data = t_orig[start_idx:, p_idx, int(md['Nx']/2):]
    tprof_mean = np.mean(t_data, axis=0)
    tprof_max = t_data[np.argmax(np.max(t_data, axis=0))]

    b_data = b_orig[start_idx:, p_idx, int(md['Nx']/2):]
    bprof_mean = np.mean(b_data, axis=0)
    bprof_max = b_data[np.argmax(np.max(b_data, axis=0))]

    ax['B'].plot(bprof_mean, gxf[int(md['Nx']/2):]-md['LX']/2, color='k')
    ax['B'].plot(bprof_max, gxf[int(md['Nx']/2):]-md['LX']/2, color='r', linestyle='--', alpha=0.5)

    ax['T'].plot(gxf[int(md['Nx']/2):]-md['LX']/2, tprof_mean, color='k')
    ax['T'].plot(gxf[int(md['Nx']/2):]-md['LX']/2, tprof_max, color='r', linestyle='--', alpha=0.5)

    ax['B'].plot(b_orig[-1, p_idx, int(md['Nx']/2):], gxf[int(md['Nx']/2):]-md['LX']/2,
            color='r')
    ax['T'].plot(gxf[int(md['Nx']/2):]-md['LX']/2, t_orig[-1, p_idx, int(md['Nx']/2):],
            color='r')

    ax['B'].set_ylim(0,0.1)
    ax['B'].set_xlim(bmin, bmax)
    ax['T'].set_xlim(0,0.1)
    ax['T'].set_ylim(tmin, tmax)

    #########################################################

    im_corrected = ax['J'].pcolormesh(sx, sy, scatter_corrected[-1], cmap=s_cmap, norm=s_norm)

    test_array = np.zeros_like(t[-1])
    for i in range(int(md['Nb'])):
        for j in range(int(md['Nphi'])):
            test_array[np.logical_and(np.logical_and(b[-1] > bbins[i] - db/2,
                b[-1] <= bbins[i] + db/2),np.logical_and(t[-1] > tbins[j] - dt/2,
                t[-1] <= tbins[j] + dt/2))] = scatter_corrected[-1,j,i]

    trac_im = ax['S'].pcolormesh(X, Y, test_array, cmap=s_cmap, norm=s_norm)
    b_cont = ax['S'].contour(Xf, Yf, np.where(t[-1] <= 5e-4, b[-1], np.NaN),
                levels = contours_b, cmap='cool', alpha=0.8)
    b_cont_fill = ax['S'].contourf(b_cont, levels = contours_b, cmap='cool', alpha=0.8, extend='min')
    t_cont = ax['S'].contour(Xf, Yf, t[-1], levels = [5e-4], colors='green', alpha=0.8)

    #########################################################

    sx_nan = sx[~np.isnan(scatter_corrected[-1])].flatten()
    sy_nan = sy[~np.isnan(scatter_corrected[-1])].flatten()
    points = np.array(list(zip(sx_nan,sy_nan)))

    if len(points) > 0:
        hull = spatial.ConvexHull(points)

        ax['J'].plot(points[hull.simplices[0],0], points[hull.simplices[0],1], 'r--', label="convex hull")
        for simplex in hull.simplices:
            ax['J'].plot(points[simplex,0], points[simplex,1], 'r--')

    #########################################################

    bgrid, tgrid = np.meshgrid(bbins, tbins)

    b_com = np.nansum(bgrid * np.where(scatter_corrected[-1] > 0, scatter_corrected[-1], 0)) \
            / np.nansum(np.where(scatter_corrected[-1]>0, scatter_corrected[-1],0))
    t_com = np.nansum(tgrid * np.where(scatter_corrected[-1] > 0, scatter_corrected[-1], 0)) \
            / np.nansum(np.where(scatter_corrected[-1]>0, scatter_corrected[-1],0))

    ax['J'].scatter(b_com, t_com, color='red', edgecolor='white', marker='^', s=200)

    b_com = np.nansum(bgrid * np.where(scatter_corrected[-1] < 0, scatter_corrected[-1], 0)) \
            / np.nansum(np.where(scatter_corrected[-1]<0, scatter_corrected[-1],0))
    t_com = np.nansum(tgrid * np.where(scatter_corrected[-1] < 0, scatter_corrected[-1], 0)) \
            / np.nansum(np.where(scatter_corrected[-1]<0, scatter_corrected[-1],0))

    ax['J'].scatter(b_com, t_com, color='blue', edgecolor='white', marker='^', s=200)

    #########################################################

    # Decorations
    im_cont_cb = plt.colorbar(b_cont_fill, ax=ax['S'], format=lambda x,_ :f"{x:.3f}", label="buoyancy")
    im_cont_cb.set_alpha(1)
    im_cont_cb.draw_all()

    corr_cb = plt.colorbar(im_corrected, ax=ax['J'], label="volume")

    ax['S'].set_title("Plume cross-section")
    ax['J'].set_title("Accumulated $(b,\\phi)$ distribution input to stratified region")

    ax['J'].set_xlabel("buoyancy")
    ax['J'].set_ylabel("tracer")
    ax['J'].set_xlim(bmin, bmax)
    ax['J'].set_ylim(tmin, tmax)

    ax['J'].axvline(b_zmax, color='k', linestyle='--')
    ax['J'].axhline(t_rms, color='k', linestyle='--')
    ax['J'].legend()

    ax['S'].set_xlim(0.2, 0.4)

    #plt.savefig(save_dir+'scatter.png',dpi=300)
    #########################################################

    def animate(step):
        global corr_cb

        s_cvals = np.concatenate((np.array([-0.5*np.nanmax(scatter_corrected[step])]),
            np.linspace(0,np.nanmax(scatter_corrected[step]),31)[:]))
        s_colors = np.concatenate((np.array([[0.,0.,1.,1.,]]),
            plt.cm.hot_r(np.linspace(0,1,31))))

        s_norm=plt.Normalize(min(s_cvals),max(s_cvals))
        s_tuples = list(zip(map(s_norm,s_cvals), s_colors))
        s_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", s_tuples, N=60)

        cvals = [-np.nanmax(scatter_flux[step]), 0, np.nanmax(scatter_flux[step])]
        colors = ["blue","white","red"]

        norm=plt.Normalize(min(cvals),max(cvals))
        tuples = list(zip(map(norm,cvals), colors))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

        ax['S'].clear()
        ax['J'].clear()

        ax['S'].set_title("Plume cross-section")
        ax['J'].set_title("Accumulated $(b,\\phi)$ distribution input to stratified region")

        ax['J'].axvline(b_zmax, color='k', linestyle='--')
        ax['J'].axhline(t_rms, color='k', linestyle='--')

        ax['B'].clear()
        ax['T'].clear()
        ax['B'].plot(bprof_mean, gxf[int(md['Nx']/2):]-md['LX']/2, color='k')
        ax['B'].plot(bprof_max, gxf[int(md['Nx']/2):]-md['LX']/2, color='r', linestyle='--', alpha=0.5)

        ax['T'].plot(gxf[int(md['Nx']/2):]-md['LX']/2, tprof_mean, color='k')
        ax['T'].plot(gxf[int(md['Nx']/2):]-md['LX']/2, tprof_max, color='r', linestyle='--', alpha=0.5)

        ax['B'].plot(b_orig[start_idx+step, p_idx, int(md['Nx']/2):], gxf[int(md['Nx']/2):]-md['LX']/2,
                color='r')
        ax['T'].plot(gxf[int(md['Nx']/2):]-md['LX']/2, t_orig[start_idx+step, p_idx, int(md['Nx']/2):],
                color='r')

        ax['B'].set_ylim(0,0.1)
        ax['B'].set_xlim(bmin, bmax)
        ax['T'].set_xlim(0,0.1)
        ax['T'].set_ylim(tmin, tmax)

        im_corrected = ax['J'].pcolormesh(sx, sy, scatter_corrected[step], cmap=s_cmap, norm=s_norm)

        corr_cb.remove()
        corr_cb = plt.colorbar(im_corrected, ax=ax['J'], label="volume")

        b_com = np.nansum(bgrid * np.where(scatter_corrected[step] > 0, scatter_corrected[step], 0)) \
                / np.nansum(np.where(scatter_corrected[step]>0, scatter_corrected[step],0))
        t_com = np.nansum(tgrid * np.where(scatter_corrected[step] > 0, scatter_corrected[step], 0)) \
                / np.nansum(np.where(scatter_corrected[step]>0, scatter_corrected[step],0))

        ax['J'].scatter(b_com, t_com, color='red', edgecolor='white', marker='^', s=200)

        b_com = np.nansum(bgrid * np.where(scatter_corrected[step] < 0, scatter_corrected[step], 0)) \
                / np.nansum(np.where(scatter_corrected[step]<0, scatter_corrected[step],0))
        t_com = np.nansum(tgrid * np.where(scatter_corrected[step] < 0, scatter_corrected[step], 0)) \
                / np.nansum(np.where(scatter_corrected[step]<0, scatter_corrected[step],0))

        ax['J'].scatter(b_com, t_com, color='blue', edgecolor='white', marker='^', s=200)

        sx_nan = sx[~np.isnan(scatter_corrected[step])].flatten()
        sy_nan = sy[~np.isnan(scatter_corrected[step])].flatten()
        points = np.array(list(zip(sx_nan,sy_nan)))

        if len(points) > 0:
            hull = spatial.ConvexHull(points)

            ax['J'].plot(points[hull.simplices[0],0], points[hull.simplices[0],1], 'r--', label="convex hull")
            for simplex in hull.simplices:
                ax['J'].plot(points[simplex,0], points[simplex,1], 'r--')

        fig.suptitle("time = {0:.2f} s".format(times[step]))

        test_array = np.zeros_like(t[step])
        for i in range(int(md['Nb'])):
            for j in range(int(md['Nphi'])):
                test_array[np.logical_and(np.logical_and(b[step] > bbins[i] - db/2,
                    b[step] <= bbins[i] + db/2),np.logical_and(t[step] > tbins[j] - dt/2,
                    t[step] <= tbins[j] + dt/2))] = scatter_corrected[step,j,i]

        trac_im = ax['S'].pcolormesh(X, Y, test_array, cmap=s_cmap, norm=s_norm)

        b_cont = ax['S'].contour(Xf, Yf, np.where(t[step] <= 5e-4, b[step], np.NaN),
                levels = contours_b, cmap='cool', alpha=0.8)
        b_cont_fill = ax['S'].contourf(b_cont, levels = contours_b, cmap='cool', alpha=0.8, extend='min')
        t_cont = ax['S'].contour(Xf, Yf, t[step], levels = [5e-4], colors='green', alpha=0.8)

        ax['S'].set_xlim(0.2, 0.4)

        ax['J'].set_xlabel("buoyancy")
        ax['J'].set_ylabel("tracer")
        ax['J'].set_xlim(bmin, bmax)
        ax['J'].set_ylim(tmin, tmax)

        ax['J'].legend()

        return im_corrected, trac_im

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=4, bitrate=-1)

    anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP, repeat=False)
    now = datetime.now()
    #anim.save(save_dir+'scatter_%s.mp4'%now.strftime("%d-%m-%Y"),writer=writer)
    plt.show()
