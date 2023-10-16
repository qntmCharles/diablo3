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

fig_save_dir = '/home/cwp29/Documents/papers/conv_pen/draft2/figs'

dbdphi = 42.2

##### ---------------------- #####

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data acquisition
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)
md['kappa'] = md['nu']/0.7

gxf, gyf, gzf, dzf = get_grid(save_dir+"/grid.h5", md)
gx, gy, gz, dz = get_grid(save_dir+"/grid.h5", md, fractional_grid=False)

print("Complete metadata: ", md)

# Get data
with h5py.File(join(save_dir, 'az_stats.h5'), 'r') as f:
    time_keys = list(f['u_az'].keys())

    # (u,v,w,b,p,phi) data
    u_az = np.array([np.array(f['u_az'][t]) for t in time_keys])
    v_az = np.array([np.array(f['v_az'][t]) for t in time_keys])
    w_az = np.array([np.array(f['w_az'][t]) for t in time_keys])
    b_az = np.array([np.array(f['b_az'][t]) for t in time_keys])
    phi_az = np.array([np.array(f['th_az'][t]) for t in time_keys])

    times = np.array([float(f['u_az'][t].attrs['Time']) for t in time_keys])
    NSAMP = len(times)

    f.close()

r_0 = md['r0']
print(r_0)
print(md['H']/r_0)
dr = md['LX']/md['Nx']
nbins = int(md['Nx']/2)
r_bins = np.array([r*dr for r in range(0, nbins+1)])
r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])

X, Y = np.meshgrid(r_bins, gz)
Xf, Yf = np.meshgrid(r_points, gzf)

var1 = w_az
var2 = phi_az

fig, axs = plt.subplots(1,2,figsize=(15, 10))
ims = np.array([None,None])
cb = np.array([None,None])

print("Setting up initial plot...")
ims[0] = axs[0].pcolormesh(X, Y, var1[-1], cmap='jet')
ims[1] = axs[1].pcolormesh(X, Y, var2[-1], cmap='jet')

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0])
cb[1] = plt.colorbar(ims[1],ax=axs[1])

#ims[0].set_clim(0, 0.03)
#ims[1].set_clim(0, 0.03)

fig.suptitle("time = 0 secs")
axs[0].set_ylabel("$z$")
axs[1].set_ylabel("$z$")
axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$y$")

axs[0].set_ylim(0, 3*md['H'])
axs[1].set_ylim(0, 3*md['H'])

#axs[0].set_xlim(0.2, 0.4)
#axs[1].set_xlim(0.2, 0.4)

axs[0].set_aspect(1)
axs[1].set_aspect(1)

def animate(step):
    ims[0].set_array(var1[step].ravel())
    ims[1].set_array(var2[step].ravel())
    fig.suptitle("time = {0:.2f} secs".format(times[step]))

    return ims.flatten(),

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'jet_%s.mp4'%now.strftime("%d-%m-%Y-%H-%m"),writer=writer)
plt.show()
