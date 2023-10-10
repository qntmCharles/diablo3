import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d, compute_F0, get_index
from os.path import join, isfile
from os import listdir
from scipy import integrate, optimize

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
dirs = ["gaussianf", "tanhf", "cosf"]

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(join(run_dir,dirs[0]), version)
gxf, gyf, gzf, dzf = get_grid(join(save_dir,dirs[0])+"/grid.h5", md)
gx, gy, gz, dz = get_grid(join(save_dir,dirs[0])+"/grid.h5", md, fractional_grid=False)

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

print("Complete metadata: ", md)

dr = md['LX']/md['Nx']
nbins = int(md['Nx']/2)
r_bins = np.array([r*dr for r in range(0, nbins+1)])
r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])

# Get data
bs = []
ts = []
ws = []
mds = []
Fs = []
scatters = []
scatter_fluxes = []
scatter_correcteds = []

NSAMP = -1
for d in dirs:
    md = get_metadata(join(run_dir,d), version)
    mds.append(md)
    with h5py.File(join(save_dir,d)+"/movie.h5", 'r') as f:
        print("Keys: %s" % f.keys())
        time_keys = list(f['th1_xz'])
        print(time_keys)
        # Get buoyancy data
        th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
        th1_xz = g2gf_1d(md,th1_xz)
        th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
        th2_xz = g2gf_1d(md,th2_xz)
        w_xz = np.array([np.array(f['w_xz'][t]) for t in time_keys])
        w_xz = g2gf_1d(md,w_xz)

        scatter = np.array([np.array(f['td_scatter'][t]) for t in time_keys])
        scatter_flux = np.array([np.array(f['td_flux'][t]) for t in time_keys])

        if NSAMP > 0:
            if len(th1_xz) != NSAMP:
                print("ERROR: mismatched simulation lengths")
        else:
            NSAMP = len(th1_xz)
        times = np.array([float(t)*md['SAVE_MOVIE_DT'] for t in time_keys])
        f.close()

    # accumulate fluxes
    for i in range(1,NSAMP):
        scatter_flux[i] += scatter_flux[i-1]

    scatter_corrected = scatter - scatter_flux

    scatter_flux = np.where(scatter_flux == 0, np.nan, scatter_flux)

    scatter_corrected = np.where(scatter_corrected == 0, np.nan, scatter_corrected)

    scatter = np.where(scatter == 0, np.nan, scatter)

    bs.append(th1_xz)
    ts.append(th2_xz)
    ws.append(w_xz)
    Fs.append(compute_F0(join(save_dir, d), md, tstart_ind = get_index(3, times), verbose=False,tracer=False))
    scatters.append(scatter)
    scatter_fluxes.append(scatter_flux)
    scatter_correcteds.append(scatter_corrected)

bs = np.array(bs)
ts = np.array(ts)
ws = np.array(ws)
mds = np.array(mds)
Fs = np.array(Fs)
scatters = np.array(scatters)
scatter_fluxes = np.array(scatter_fluxes)

print(Fs)
print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

#############################################################################################################

bmin = 0
bmax = np.array([mds[i]['b_factor']*mds[i]['N2']*(mds[i]['LZ']-mds[i]['H']) for i in range(len(dirs))])

F0 = np.array([mds[i]['b0']*(mds[i]['r0']**2) for i in range(len(dirs))])
alpha = mds[0]['alpha_e']

tmin = 5e-4
tmax = np.array([mds[i]['phi_factor']*5*F0[i] / (3 * alpha) * np.power(0.9*alpha*F0[i], -1/3) * np.power(
        mds[i]['H']+ 5*mds[i]['r0']/(6*alpha), -5/3) for i in range(len(dirs))])

Nb = int(mds[0]['Nb'])
Nt = int(mds[0]['Nphi'])
db = np.array([round((bmax[i] - bmin)/Nb,3) for i in range(len(dirs))])
dt = np.array([(tmax[i] - tmin)/Nt for i in range(len(dirs))])
dx = md['LX']/md['Nx']
dy = md['LY']/md['Ny']
bbins = np.array([[bmin + (i+0.5)*db[j] for i in range(Nb)] for j in range(len(dirs))])
tbins = np.array([[tmin + (i+0.5)*dt[j] for i in range(Nt)] for j in range(len(dirs))])

#############################################################################################################

bf_dict = {}
wf_dict = {}

for d in range(len(dirs)):
    alpha = mds[d]['alpha_e']
    zvirt = -5/6 * mds[d]['r0']/alpha
    F0 = mds[d]['r0']**2 * mds[d]['b0']

    r_m = np.zeros(shape=(md['Nz'],))
    b_m = np.zeros(shape=(md['Nz'],))
    w_m = np.zeros(shape=(md['Nz'],))

    for j in range(md['Nz']):
        r_m[j] = 1.2 * alpha * (gzf[j] - zvirt)
        w_m[j] = (0.9 * alpha * F0)**(1/3) * (gzf[j]-zvirt)**(2/3)/r_m[j]
        b_m[j] = F0/(r_m[j] * r_m[j] * w_m[j])

    b_forcing = np.zeros(shape=(md['Nz'],md['Nx']+1))
    w_forcing = np.zeros(shape=(md['Nz'],md['Nx']+1))

    for j in range(md['Nz']):
        if mds[d]['F_TYPE'] == 7:
            w_forcing[j] = 0.2 * w_m[j] * (np.tanh((gx - md['LX']/2+4*r_m[j])/1e-3) \
                                    -np.tanh((gx - md['LX']/2-4*r_m[j])/1e-3)) \
                                    *(1 - np.tanh((gzf[j]-md['Lyc'])/md['Lyp']))/2
            b_forcing[j] = 0.2 * b_m[j] * (np.tanh((gx - md['LX']/2+4*r_m[j])/1e-3) \
                                    -np.tanh((gx - md['LX']/2-4*r_m[j])/1e-3)) \
                                    *(1 - np.tanh((gzf[j]-md['Lyc'])/md['Lyp']))/2
        elif mds[d]['F_TYPE'] == 8:
            w_forcing[j] = 2*w_m[j]*np.exp(-(gx-md['LX']/2)**2/(2*r_m[j]**2)) \
                                    * np.cos((gx-md['LX']/2)/r_m[j])**2 \
                                    * (1 - np.tanh((gzf[j]-md['Lyc'])/md['Lyp']))/2
            b_forcing[j] = 2*b_m[j]*np.exp(-(gx-md['LX']/2)**2/(2*r_m[j]**2)) \
                                    * np.cos((gx-md['LX']/2)/r_m[j])**2 \
                                    * (1 - np.tanh((gzf[j]-md['Lyc'])/md['Lyp']))/2
        else:
            w_forcing[j] = 2*w_m[j]*np.exp(-(gx-md['LX']/2)**2/(2*r_m[j]**2)) \
                                    * (1 - np.tanh((gzf[j]-md['Lyc'])/md['Lyp']))/2
            b_forcing[j] = 2*b_m[j]*np.exp(-(gx-md['LX']/2)**2/(2*r_m[j]**2)) \
                                    * (1 - np.tanh((gzf[j]-md['Lyc'])/md['Lyp']))/2

    wf_dict[dirs[d][:-1]] = w_forcing
    bf_dict[dirs[d][:-1]] = b_forcing

#############################################################################################################
# Comparison plot of tracer profiles
#############################################################################################################

strat_idx = get_index(0.8*md['H'], gzf)

summary_fig, ax = plt.subplots(2,len(dirs))

tstart_idx = get_index(3, times)
cols = plt.cm.rainbow(np.linspace(0,1,NSAMP-tstart_idx))

z_ind = 5

print(ts.shape)
for d in range(len(dirs)):
    for i, c in zip(range(tstart_idx,NSAMP),cols):
        ax[0,d].plot(gxf, ts[d, i, strat_idx, :], color=c, alpha=0.3)

    ax[0,d].plot(gxf, np.mean(ts[d, tstart_idx:, strat_idx, :], axis=0), color='k', linestyle='--')

    for i, c in zip(range(2,10),cols):
        ax[1,d].plot(gxf, ts[d, i, z_ind, :], color=c, alpha=0.3)

    ax[1,d].plot(gxf, np.mean(ts[d, 2:10, z_ind, :], axis=0), color='k', linestyle='--')
    ax[1,d].plot(gx, bf_dict[dirs[d][:-1]][z_ind], color='r', linestyle='--')

    ax[0,d].set_title(dirs[d][:-1])
    ax[0,d].set_xlim(0.2, 0.4)
    ax[1,d].set_xlim(0.2, 0.4)

    ax[1,d].plot(gxf, np.mean(ts[d, 2:10, z_ind, :], axis=0), color='k', linestyle='--')

comp_fig = plt.figure()
cols = plt.cm.rainbow(np.linspace(0,1,len(dirs)))
for d,c in zip(range(len(dirs)), cols):
    tmean = np.mean(ts[d, tstart_idx:, strat_idx, :],axis=0)
    plt.plot(gxf, tmean/np.max(tmean), color=c, label=dirs[d][:-1])

plt.xlim(0.2, 0.4)
plt.legend()


plt.tight_layout()
#plt.show()

#############################################################################################################
# Comparison plot of (t,b) distributions

fig, ax = plt.subplots(len(dirs), 3, figsize=(15,5))
fig.suptitle("time = 0.00 s")


#########################################################
# Create colour maps

cvals = [-1e-5, 0, 1e-5]
colors = ["blue","white","red"]

norm=plt.Normalize(min(cvals),max(cvals))
tuples = list(zip(map(norm,cvals), colors))
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", tuples)

s_cvals = np.concatenate((np.array([-1e-5, 0]), np.linspace(0,2e-5,31)[1:]))
s_colors = np.concatenate((np.array([[0.,0.,1.,1.,],[1.,1.,1.,1.]]), plt.cm.hot_r(np.linspace(0,1,30))))

s_norm=plt.Normalize(min(s_cvals),max(s_cvals))
s_tuples = list(zip(map(s_norm,s_cvals), s_colors))
s_cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", s_tuples)

#########################################################
# Figure set-up

# Uncomment below to scale plots so that max tracer is at the top of the plot (removes whitespace)
#tmax = [tbins[i,np.max(np.argwhere(scatters[i]>0)[:,2])] for i in range(len(dirs))]
#print(tmax)

for d in range(len(dirs)):
    sx, sy = np.meshgrid(bbins[d], tbins[d])

    im_scatter = ax[0,d].pcolormesh(sx, sy, scatters[d][-1], cmap='jet')
    im_flux = ax[1,d].pcolormesh(sx, sy, scatter_fluxes[d][-1], cmap=cmap, norm=norm)
    im_corrected = ax[2,d].pcolormesh(sx, sy, scatter_correcteds[d][-1], cmap=s_cmap, norm=s_norm)

    sc_div = make_axes_locatable(ax[0,d])
    sc_cax = sc_div.append_axes("right", size="5%", pad=0.05)
    sc_cb = plt.colorbar(im_scatter, cax=sc_cax, label="volume")

    flux_div = make_axes_locatable(ax[1,d])
    flux_cax = flux_div.append_axes("right", size="5%", pad=0.05)
    flux_cb = plt.colorbar(im_flux, cax=flux_cax, label="volume")

    corr_div = make_axes_locatable(ax[2,d])
    corr_cax = corr_div.append_axes("right", size="5%", pad=0.05)
    corr_cb = plt.colorbar(im_corrected, cax=corr_cax, label="volume")

    ax[0,d].set_title(dirs[d][:-1])
    ax[0,d].set_xlabel("buoyancy")
    ax[0,d].set_ylabel("tracer")
    ax[0,d].set_xlim(bmin, 0.5*bmax[d])
    ax[0,d].set_ylim(tmin, tmax[d])

    ax[1,d].set_title(dirs[d][:-1])
    ax[1,d].set_xlabel("buoyancy")
    ax[1,d].set_ylabel("tracer")
    ax[1,d].set_xlim(bmin, 0.5*bmax[d])
    ax[1,d].set_ylim(tmin, tmax[d])

    ax[2,d].set_title(dirs[d][:-1])
    ax[2,d].set_xlabel("buoyancy")
    ax[2,d].set_ylabel("tracer")
    ax[2,d].set_xlim(bmin, 0.5*bmax[d])
    ax[2,d].set_ylim(tmin, tmax[d])

#plt.tight_layout()
#plt.show()
#########################################################
def animate(step):
    for d in range(len(dirs)):
        ax[0,d].clear()
        ax[1,d].clear()
        ax[2,d].clear()

        sx, sy = np.meshgrid(bbins[d], tbins[d])

        im_scatter = ax[0,d].pcolormesh(sx, sy, scatters[d][step], cmap='jet')
        im_flux = ax[1,d].pcolormesh(sx, sy, scatter_fluxes[d][step], cmap=cmap, norm=norm)
        im_corrected = ax[2,d].pcolormesh(sx, sy, scatter_correcteds[d][step], cmap=s_cmap, norm=s_norm)

        ax[0,d].set_title(dirs[d][:-1])
        ax[0,d].set_xlabel("buoyancy")
        ax[0,d].set_ylabel("tracer")
        ax[0,d].set_xlim(bmin, 0.5*bmax[d])
        ax[0,d].set_ylim(tmin, tmax[d])

        ax[1,d].set_title(dirs[d][:-1])
        ax[1,d].set_xlabel("buoyancy")
        ax[1,d].set_ylabel("tracer")
        ax[1,d].set_xlim(bmin, 0.5*bmax[d])
        ax[1,d].set_ylim(tmin, tmax[d])

        ax[2,d].set_title(dirs[d][:-1])
        ax[2,d].set_xlabel("buoyancy")
        ax[2,d].set_ylabel("tracer")
        ax[2,d].set_xlim(bmin, 0.5*bmax[d])
        ax[2,d].set_ylim(tmin, tmax[d])

    return im_scatter, im_flux, im_corrected,

Writer = animation.writers['ffmpeg']
writer = Writer(fps=4, bitrate=-1)

anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)

plt.show()
