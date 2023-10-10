# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py, cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d
from scipy import ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dz = get_grid(save_dir+"/grid.h5", md)
z_coords, x_coords = np.meshgrid(gzf, gxf, indexing='ij')

print("Complete metadata: ", md)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Movie keys: %s" % f.keys())
    time_keys = list(f['th1_xy'])
    print(time_keys)
    # Get buoyancy data
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    w_xz = np.array([np.array(f['w_xz'][t]) for t in time_keys])
    th1_xz = g2gf_1d(th1_xz)
    w_xz = g2gf_1d(w_xz)
    NSAMP = len(th1_xz)
    times = np.array([float(t)*md['SAVE_MOVIE_DT'] for t in time_keys])
    f.close()

with h5py.File(save_dir+"/mean.h5", 'r') as f:
    print("Mean keys: %s" % f.keys())
    m_time_keys = list(f['wrms'])
    print(m_time_keys)
    w_rms = np.array([np.array(f['wrms'][t]) for t in time_keys])
    u_rms = np.array([np.array(f['urms'][t]) for t in time_keys])
    v_rms = np.array([np.array(f['vrms'][t]) for t in time_keys])

    u_rms = g2gf_1d(u_rms)
    v_rms = g2gf_1d(v_rms)
    w_rms = g2gf_1d(w_rms)
    f.close()

th1_xz = np.flip(th1_xz,axis=1)
w_xz = np.flip(w_xz,axis=1)
N2_xz = np.gradient(th1_xz, gzf, axis=1)
N2t_xz = np.gradient(N2_xz, times, axis=0)

# (attempted) Low-pass filter on w_xz
w_xz_filtered = np.zeros_like(w_xz)
th1_xz_filtered = np.zeros_like(th1_xz)

for i in range(NSAMP):
    w_xz_filtered[i] = ndimage.gaussian_filter(w_xz[i],2)
    th1_xz_filtered[i] = ndimage.gaussian_filter(th1_xz[i],2)


N2_filtered = np.gradient(th1_xz_filtered, gzf, axis=1)
N2t_xz_filtered = np.gradient(np.gradient(th1_xz_filtered, gzf, axis=1), times, axis=0)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

##### Plot vertical profile of N^2 #####
fig, ax = plt.subplots(1,4)

xplot = int(md['Nx']/2)

line1, = ax[0].plot(np.flip(N2_filtered[0][:,xplot],axis=0), gzf, color='b')
line2, = ax[1].plot(np.flip(th1_xz_filtered[0][:,xplot],axis=0), gzf, color='b')
line3, = ax[2].plot(w_rms[0], gzf, color='b')
line4, = ax[3].plot(u_rms[0], gzf, color='b')

fig.suptitle("time = 0 s")
ax[0].set_xlabel("$N^2$")
ax[0].set_ylabel("z")
#ax[0].set_xlim(-1e-4, 1e-4)

ax[1].set_xlabel("b")
ax[1].set_ylabel("z")

ax[2].set_xlabel("w_rms")
ax[2].set_ylabel("z")
ax[2].set_xlim(0, 5e-4)

ax[3].set_xlabel("u_rms")
ax[3].set_ylabel("z")
ax[3].set_xlim(0, 5e-4)

def animate(step):
    line1.set_xdata(np.flip(N2_filtered[step][:,xplot],axis=0))
    line2.set_xdata(np.flip(th1_xz_filtered[step][:,xplot],axis=0))
    line3.set_xdata(w_rms[step])
    line4.set_xdata(u_rms[step])
    fig.suptitle("time = {0:.4f} s".format(step*md['SAVE_STATS_DT']))

    return line1, line2, line3, line4

anim = animation.FuncAnimation(fig, animate, interval=500, frames=NSAMP)

plt.show()
