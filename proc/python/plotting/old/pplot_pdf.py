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
from functions import get_metadata, read_params, get_grid, g2gf_1d

##### USER-DEFINED PARAMETERS #####
params_file = "params.dat"
out_file = "out.002249.h5"
mp = 12

save = False

##### ----------------------- #####

##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(run_dir, version)
print("Complete metadata: ",md)

# Calculate turnover time
F = md['b0'] * (md['r0']**2)
tau = md['r0']**(4/3) * F**(-1/3)
print("Non-dimensional turnover time: {0:.04f}".format(tau))

##### Get grid #####
gxf, gyf, gzf, dzf = get_grid(join(run_dir, 'grid.h5'), md)
gx, gy, gz, dz = get_grid(join(run_dir, 'grid.h5'), md, fractional_grid=False)

##### Get data #####
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    b = np.array([np.array(f['th1_xz'][tstep]) for tstep in time_keys])
    t = np.array([np.array(f['th2_xz'][tstep]) for tstep in time_keys])
    b = g2gf_1d(md, b)
    t = g2gf_1d(md, t)
    NSAMP = len(b)
    #times = np.array([float(tstep)*md['SAVE_MOVIE_DT'] for tstep in time_keys])
    times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

# Compute time indices
nplot = 3
interval = 100

step = np.round(interval*tau / md['SAVE_STATS_DT'])

t_inds = list(map(int,step*np.array(range(1, nplot+1))))
tplot = [times[i] for i in t_inds]
print("Plotting at times: ",tplot)
print("with interval", step*md['SAVE_STATS_DT'])

##### Compute PDF #####
# Create buoyancy bins
b_min = 0
b_max = np.max(b[0,:,int(md['Nx']/2)])/4
nbins = 129

check_data = b[0,:,int(md['Nx']/2)]
plot_min = -1
plot_max = -1
for j in range(md['Nz']):
    if gzf[j] < md['H'] and gzf[j+1] > md['H']:
        plot_min = j
    if check_data[j-1] <= b_max and check_data[j] >= b_max:
        plot_max = j

if plot_min == -1: print("Warning: plot_min miscalculated")
if plot_max == -1: print("Warning: plot_max miscalculated")

bbins, db = np.linspace(b_min, b_max, nbins, retstep=True)
bins = [0 for i in range(nbins)]

for i in range(nbins-1):
    bins[i] = np.sum(np.where(np.logical_and(b[0][plot_min:plot_max] > bbins[i],
        b[0][plot_min:plot_max] < bbins[i+1]), t[0][plot_min:plot_max], 0))

# Normalise pdf
area = integrate.trapezoid(np.abs(bins), dx = db)

##### Set up plot #####
b_cont = 0.1
t_cont = 0.005
isopycnals = np.linspace(b_min, b_max, 10)[1:-1]

cols = plt.cm.Greys(np.linspace(0,1,nplot+1))
tcols = plt.cm.OrRd(np.linspace(0,1,nplot+1))[1:]

X, Y = np.meshgrid(gx, gz[plot_min:plot_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[plot_min:plot_max])

fig, ax = plt.subplots(1, 2, figsize=(12,5))

b_bg = ax[0].pcolormesh(X, Y, b[0][plot_min:plot_max], cmap=plt.cm.get_cmap('jet'), alpha=0.3)

for step,c,d in zip(t_inds, cols[1:],tcols):
    #ax[0].contour(Xf, Yf, b[step][plot_min:plot_max], levels=isopycnals, colors=[c], alpha=0.3)
    ax[0].contour(Xf, Yf, t[step][plot_min:plot_max], levels=[t_cont], colors=[d], linestyles='dashed')

    bins = [0 for i in range(nbins)]

    for i in range(nbins-1):
        bins[i] = np.sum(np.where(np.logical_and(b[step][plot_min:plot_max] > bbins[i],
            b[step][plot_min:plot_max] < bbins[i+1]), t[step][plot_min:plot_max], 0))

    #area = integrate.trapezoid(np.abs(bins), dx = db)
    ax[1].plot(np.array(bins), bbins, color=d, label = "t={0:.3f} s".format(times[step]))

rs = np.linspace(0, md['LX']/2, 100)
rm = 1.2 * md['alpha_e'] * md['H']
alpha = md['alpha_e']
#theoretical_pdf = md['H'] + (np.exp(-0.5 * np.power(rs,2) / (6/5 * alpha * md['H'])**2) * 5* F * np.power(
        #9/10 * alpha * F,-1/3) * np.power(md['H'],-5/3) / (3*alpha*md['N2']))
theoretical_pdf = (rs/rm)*np.exp(-0.5*np.power(rs/rm,2))
#theoretical_pdf /= integrate.trapezoid(theoretical_pdf, rs/rm)
bs = (np.exp(-0.5 * np.power(rs,2) / (6/5 * alpha * md['H'])**2) * 5* F * np.power(
        9/10 * alpha * F,-1/3) * np.power(md['H'],-5/3) / (3*alpha))

ax[1].plot(theoretical_pdf, bs)


ax[1].set_ylim(b_min, b_max)
ax[1].set_xlim(-0.05, 3)
ax[1].axvline(0, color='k', linestyle='--', alpha=0.5)
ax[1].legend()
ax[0].set_xlabel("x")
ax[0].set_ylabel("z")
ax[1].set_ylabel("buoyancy")
ax[1].set_xlabel("probability density")


plt.tight_layout()
plt.show()
