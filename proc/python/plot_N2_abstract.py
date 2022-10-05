# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py, cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid
from scipy import ndimage

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dzf = get_grid(save_dir+"/grid.h5", md)
gx, gy, gz, dz = get_grid(save_dir+"/grid.h5", md, fractional_grid=False)
z_coords, x_coords = np.meshgrid(gzf, gxf, indexing='ij')

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

print("Complete metadata: ", md)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xy'])
    print(time_keys)
    # Get buoyancy data
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    NSAMP = len(th1_xz)
    times = np.array([(float(t)-1)*md['SAVE_MOVIE_DT'] for t in time_keys])
    f.close()

N2_xz = np.gradient(th1_xz, gzf, axis=1)
N2t_xz = np.gradient(N2_xz, times, axis=0)

# Low-pass filter
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

contour_lvls = np.linspace(0.005,0.05, 5)

print("Setting up initial plot...")
ims[0] = axs[0].pcolormesh(X,Y,th1_xz_filtered[0], cmap=plt.cm.get_cmap('bwr'))
ims[1] = axs[1].pcolormesh(X,Y,N2t_xz_filtered[0], cmap=plt.cm.get_cmap('jet'))
c = axs[0].contour(Xf, Yf, th2_xz[0], levels=contour_lvls, colors='grey', alpha=0.5)

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0])
cb[1] = plt.colorbar(ims[1],ax=axs[1])
ims[0].set_clim(-0.1, 0.1)
ims[1].set_clim(-0.2, 0.2)

fig.suptitle("time = 0.0000 secs, t step = 0")
axs[0].set_ylabel("$z$")
axs[1].set_ylabel("$z$")
axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$y$")

def animate(step):
    global c

    ims[0].set_array((th1_xz_filtered[step]-th1_xz_filtered[0]).ravel()) #perturbation
    ims[1].set_array(N2t_xz_filtered[step].ravel())
    fig.suptitle("time = {0:.4f} secs, t step = {1}".format(times[step], step))

    for coll in c.collections:
        coll.remove()
    c = axs[0].contour(Xf, Yf, th2_xz[step], levels=contour_lvls, colors='grey', alpha=0.5)

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
tstep = 47#int(input("Time step to show: "))
"""
fig, axs = plt.subplots(1,2,figsize=(18, 6))
ims = np.array([None,None])
cb = np.array([None,None])

print("Setting up initial plot...")
ims[0] = axs[0].pcolormesh(X,Y,th1_xz_filtered[tstep]-th1_xz_filtered[0], cmap=plt.cm.get_cmap('bwr'))
ims[1] = axs[1].pcolormesh(X,Y,N2t_xz_filtered[tstep], cmap=plt.cm.get_cmap('jet'))
c = axs[0].contour(Xf, Yf, th2_xz[0], levels=contour_lvls, colors='grey', alpha=0.5)

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0], label="$b \, (m \cdot s^{{-2}})$")
cb[1] = plt.colorbar(ims[1],ax=axs[1], label="$\\partial_t N^2 \, (s^{{-3}})$")
ims[0].set_clim(-0.1, 0.1)
ims[1].set_clim(-0.2, 0.2)

axs[0].set_ylabel("$z \, (m)$")
axs[1].set_ylabel("$z \, (m)$")
axs[0].set_xlabel("$x \, (m)$")
axs[1].set_xlabel("$x \, (m)$")
"""

N = np.sqrt(md['N2'])
F0 = md['b0'] * np.power(md['r0'],2)
alpha = 0.1

t_nd = np.power(N, -1)
L_nd = np.power(N, -3/4) * np.power(F0, 1/4)

xstart = 0.1
xstop = 0.5
zstart = 0
zstop = 0.4

xi = get_index(xstart, gx)
xf = get_index(xstop, gx)
zi = get_index(zstart, gz)
zf = get_index(zstop, gz)

#fig = plt.figure(figsize=(6.5, 6),facecolor=(0.9,0.9,0.9))
fig = plt.figure(figsize=(6.5, 6))
ax = plt.gca()

#ax.set_facecolor((0.9,0.9,0.9))

gray = (0.9,0.9,0.9)
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue", 'white', "red"])

print(th1_xz_filtered.shape)

im = plt.pcolormesh(X[zi:zf,xi:xf]/L_nd,Y[zi:zf,xi:xf]/L_nd,
        th1_xz_filtered[tstep, zi:zf, xi:xf]-th1_xz_filtered[0, zi:zf, xi:xf], cmap=cmap)
c_step = plt.contour(Xf[zi:zf,xi:xf]/L_nd, Yf[zi:zf,xi:xf]/L_nd,
        th2_xz[tstep,zi:zf,xi:xf], levels=contour_lvls, colors='k', alpha=0.5)

zmax_dim = 1.36 * np.power(alpha, -1/2) * np.power(F0, 1/4) * np.power(N, -3/4)

wm = np.power(0.9 * alpha * F0, 1/3) * np.power(md['H'], -1/3) / (1.2 * alpha)
zmax_en = wm * 2 * np.power(N, -1)

zmax_num = 0.2454

plt.axhline((md['H']+zmax_dim)/L_nd, color='black', linestyle=':')
plt.text((xstart+0.004)/L_nd, 0.2+(md['H']+zmax_dim)/L_nd, "Dimensional analysis", fontsize=13)

plt.axhline((md['H']+zmax_en)/L_nd, color='black', linestyle=':')
plt.text((xstart+0.004)/L_nd, 0.2+(md['H']+zmax_en)/L_nd, "Energetic", fontsize=13)

plt.axhline((zmax_num)/L_nd, color='black', linestyle=':')
plt.text((xstart+0.004)/L_nd, -0.8+(zmax_num)/L_nd, "Plume equations", fontsize=13)

plt.axhline((md['Lyc']+md['Lyp'])/L_nd, color='black', linestyle='--')
plt.text((xstart+0.004)/L_nd, (-0.013+md['Lyc']+md['Lyp'])/L_nd, "Forcing", fontsize=13)

plt.xlabel("x (non-dimensional)")
plt.ylabel("z (non-dimensional)")

im.set_clim(-0.05, 0.05)

plt.tight_layout()
plt.savefig('/home/cwp29/Documents/posters/issf2/ansong.png', dpi=300)
plt.show()
