import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(save_dir+'grid.h5', md)
gx, gy, gz, dz = get_grid(save_dir+'grid.h5', md, fractional_grid=False)

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

print("Complete metadata: ", md)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xy'])
    print(time_keys)
    # Get buoyancy data
    b = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    phi_wv = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    phi_c = np.array([np.array(f['th3_xz'][t]) for t in time_keys])

    NSAMP = len(b)
    times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

##### ---------------------- #####

alpha = md['alpha']
beta = md['beta']
q_0 = md['q0']

T = b - beta * Yf
phi_sat = q_0 * np.exp(alpha * T)

#phi_c = np.where(phi_wv > phi_sat, phi_wv - phi_sat, 0)
#phi_c = phi_sat

##### ---------------------- #####


print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
fig, axs = plt.subplots(1,3,figsize=(15, 5))
ims = np.array([None,None,None])
cb = np.array([None,None,None])

print("Setting up initial plot...")
ims[0] = axs[0].pcolormesh(X, Y, b[-1], cmap='jet')
ims[1] = axs[1].pcolormesh(X, Y, phi_wv[-1], cmap='jet')
ims[2] = axs[2].pcolormesh(X, Y, phi_c[-1], cmap='jet')

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[2].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0])
cb[1] = plt.colorbar(ims[1],ax=axs[1])
cb[2] = plt.colorbar(ims[2],ax=axs[2])

ims[0].set_clim(0, 0.03)
ims[1].set_clim(0, 0.03)
ims[2].set_clim(0, 0.03)

fig.suptitle("time = 0 secs")
axs[0].set_ylabel("$z$")
axs[1].set_ylabel("$z$")
axs[2].set_ylabel("$z$")

axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$x$")
axs[2].set_xlabel("$x$")

axs[0].set_ylim(0, 0.4)
axs[1].set_ylim(0, 0.4)
axs[2].set_ylim(0, 0.4)
axs[0].set_xlim(0.2, 0.4)
axs[1].set_xlim(0.2, 0.4)
axs[2].set_xlim(0.2, 0.4)

axs[0].set_aspect(1)
axs[1].set_aspect(1)
axs[2].set_aspect(1)

def animate(step):
    ims[0].set_array(b[step].ravel())
    ims[1].set_array(phi_wv[step].ravel())
    ims[2].set_array(phi_c[step].ravel())
    fig.suptitle("time = {0:.2f} secs".format(times[step]))

    return ims.flatten(),

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=500, frames=NSAMP)
now = datetime.now()
plt.show()
