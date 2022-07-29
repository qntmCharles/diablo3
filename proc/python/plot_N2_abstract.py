# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py, cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid
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
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xy'])
    print(time_keys)
    # Get buoyancy data
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    w_xz = np.array([np.array(f['w_xz'][t]) for t in time_keys])
    NSAMP = len(th1_xz)
    times = np.array([float(t)*md['SAVE_MOVIE_DT'] for t in time_keys])
    f.close()

th1_xz = np.flip(th1_xz,axis=1)
N2_xz = np.gradient(th1_xz, gzf, axis=1)
N2t_xz = np.gradient(N2_xz, times, axis=0)

# (attempted) Low-pass filter on w_xz
th1_xz_filtered = np.zeros_like(th1_xz)

for i in range(NSAMP):
    th1_xz_filtered[i] = ndimage.gaussian_filter(th1_xz[i],2)


N2t_xz_filtered = np.gradient(np.gradient(th1_xz_filtered, gzf, axis=1), times, axis=0)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
fig, axs = plt.subplots(1,2,figsize=(10, 6))
ims = np.array([None,None])
cb = np.array([None,None])

contour_lvls = np.linspace(0.01,0.1, 6)

print("Setting up initial plot...")
ims[0] = axs[0].imshow(th1_xz[0], cmap='jet', interpolation='bicubic', animated=True,
        extent=[0,md['LX'],0,md['LZ']])
ims[1] = axs[1].imshow(N2t_xz_filtered[0], cmap='jet', interpolation='bicubic', animated=True,
        extent=[0,md['LY'],0,md['LZ']])
c = axs[0].contour(np.flip(th1_xz[0],axis=0), extent=[0,md['LY'],0,md['LZ']], levels=contour_lvls,
        colors='white')

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0])
cb[1] = plt.colorbar(ims[1],ax=axs[1])
ims[0].set_clim(0, np.max(th1_xz[0]))
ims[1].set_clim(-0.2, 0.2)

fig.suptitle("time = 0.0000 secs, t step = 0")
axs[0].set_ylabel("$z$")
axs[1].set_ylabel("$z$")
axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$y$")

def animate(step):
    global c

    ims[0].set_data(th1_xz[step])
    ims[1].set_data(N2t_xz_filtered[step])
    fig.suptitle("time = {0:.4f} secs, t step = {1}".format(times[step], step))

    for coll in c.collections:
        coll.remove()
    c = axs[0].contour(np.flip(th1_xz[step],axis=0), extent=[0,md['LY'],0,md['LZ']], levels=contour_lvls,
            colors='white')

    return ims.flatten(),c,

fig.tight_layout()

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=20, frames=NSAMP)

now = datetime.now()
#anim.save(save_dir+'fountain_w_%s.mp4'%now.strftime("%Y-%m-%d:%H"),writer=writer, dpi=300)
plt.show()

tstep = int(input("Time step to show: "))
fig, axs = plt.subplots(1,2,figsize=(18, 6))
ims = np.array([None,None])
cb = np.array([None,None])

contour_lvls = np.linspace(0.01,0.1, 6)

print("Setting up initial plot...")
ims[0] = axs[0].imshow(th1_xz[tstep], cmap='jet', interpolation='bicubic', animated=True,
        extent=[0,md['LX'],0,md['LZ']])
ims[1] = axs[1].imshow(N2t_xz_filtered[tstep], cmap='jet', interpolation='bicubic', animated=True,
        extent=[0,md['LY'],0,md['LZ']])
c = axs[0].contour(np.flip(th1_xz[tstep],axis=0), extent=[0,md['LY'],0,md['LZ']], levels=contour_lvls,
        colors='white')

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0], label="$b \, (m \cdot s^{{-2}})$")
cb[1] = plt.colorbar(ims[1],ax=axs[1], label="$\\partial_t N^2 \, (s^{{-3}})$")
ims[0].set_clim(0, np.max(th1_xz[tstep]))
ims[1].set_clim(-0.2, 0.2)

#fig.suptitle("time = {0:.4f} secs, t step = {1}".format(times[tstep], tstep))
#axs[0].set_title("b")
#axs[1].set_title("$\\partial_t N^2$")
axs[0].set_ylabel("$z \, (m)$")
axs[1].set_ylabel("$z \, (m)$")
axs[0].set_xlabel("$x \, (m)$")
axs[1].set_xlabel("$x \, (m)$")

plt.tight_layout()
plt.savefig('/home/cwp29/Documents/ISSF_2022_latex/ansong.png', dpi=300)
plt.show()
