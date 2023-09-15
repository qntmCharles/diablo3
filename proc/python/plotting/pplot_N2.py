import sys, os
sys.path.insert(1, os.path.join(sys.path[0],".."))
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

gxf, gyf, gzf, dzf = get_grid(save_dir+"/grid.h5", md)
gx, gy, gz, dz = get_grid(save_dir+"/grid.h5", md, fractional_grid=False)

X, Y = np.meshgrid(gx, gz)
Xf, Yf = np.meshgrid(gxf, gzf)

print("Complete metadata: ", md)

# Get data
with h5py.File(save_dir+"/movie.h5", 'r') as f:
    print("Keys: %s" % f.keys())
    time_keys = list(f['th1_xz'])
    print(time_keys)
    # Get buoyancy data
    th1_xz = np.array([np.array(f['th1_xz'][t]) for t in time_keys])
    w_xz = np.array([np.array(f['w_xz'][t]) for t in time_keys])

    th1_xz = g2gf_1d(md, th1_xz)
    w_xz = g2gf_1d(md, w_xz)

    NSAMP = len(th1_xz)
    times = np.array([float(t)*md['SAVE_MOVIE_DT'] for t in time_keys])
    f.close()

#th1_xz = np.flip(th1_xz,axis=1)
#w_xz = np.flip(w_xz,axis=1)
N2_xz = np.gradient(th1_xz, gzf, axis=1)
N2t_xz = np.gradient(N2_xz, times, axis=0)

# Low-pass filter on w_xz
w_xz_filtered = np.zeros_like(w_xz)
th1_xz_filtered = np.zeros_like(th1_xz)

for i in range(NSAMP):
    w_xz_filtered[i] = ndimage.gaussian_filter(w_xz[i],2)
    th1_xz_filtered[i] = ndimage.gaussian_filter(th1_xz[i],2)

N2_filtered = np.gradient(th1_xz_filtered, gzf, axis=1)
N2t_xz_filtered = np.gradient(np.gradient(th1_xz_filtered, gzf, axis=1), times, axis=0)

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
fig, axs = plt.subplots(1,3,figsize=(15, 6))
ims = np.array([None,None,None])
cb = np.array([None,None,None])

contour_isopycnals = np.linspace(0,np.max(th1_xz[0]), 30)[1:]
contour_plume = np.linspace(0.02, 0.1, 4)[:-1]

print("Setting up initial plot...")
ims[0] = axs[0].pcolormesh(X,Y,th1_xz[0], cmap=plt.cm.get_cmap('bwr'))
ims[1] = axs[1].pcolormesh(X,Y,w_xz_filtered[0], cmap=plt.cm.get_cmap('jet'))
ims[2] = axs[2].pcolormesh(X,Y,N2t_xz_filtered[0], cmap=plt.cm.get_cmap('jet'))
c_isopycnal = axs[0].contour(Xf,Yf,th1_xz[0], levels=contour_isopycnals, colors='grey', alpha=0.5)
c_plume = axs[0].contour(Xf,Yf,th1_xz[0], levels=contour_plume, colors='grey',alpha=0.5)

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='black', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[2].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0])
cb[1] = plt.colorbar(ims[1],ax=axs[1])
cb[2] = plt.colorbar(ims[2],ax=axs[2])
ims[0].set_clim(-0.03,0.03)
ims[1].set_clim(-0.002,0.002)
#ims[1].set_clim(-0.0002,0.0002)
#ims[2].set_clim(-np.max(np.abs(N2t_xz[0])), np.max(np.abs(N2t_xz[0])))
ims[2].set_clim(-0.1,0.1)
#ims[2].set_clim(-0.01,0.01)

fig.suptitle("time = 0.0000 secs")
axs[0].set_title("$b$ perturbation")
axs[1].set_title("$w = \Delta z_t$")
axs[2].set_title("$N^2_t$")
axs[0].set_ylabel("$z$")
axs[1].set_ylabel("$z$")
axs[2].set_ylabel("$z$")
axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$y$")
axs[2].set_xlabel("$y$")

def animate(step):
    global c_plume, c_isopycnal

    ims[0].set_array((th1_xz[step]-th1_xz[0]).ravel()) #perturbation
    ims[1].set_array(w_xz_filtered[step].ravel())
    ims[2].set_array(N2t_xz_filtered[step].ravel())
    fig.suptitle("time = {0:.4f} secs".format(times[step]))

    for coll in c_plume.collections:
        coll.remove()
    for coll in c_isopycnal.collections:
        coll.remove()

    c_isopycnal = axs[0].contour(Xf,Yf,th1_xz[step], levels=contour_isopycnals, colors='grey', alpha=0.5)
    c_plume = axs[0].contour(Xf,Yf,th1_xz[step], levels=contour_plume, colors='grey',alpha=0.5)

    return ims.flatten(),

fig.tight_layout()

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=250, frames=NSAMP)

now = datetime.now()
#anim.save(save_dir+'fountain_w_%s.mp4'%now.strftime("%Y-%m-%d:%H"),writer=writer, dpi=300)

##### Plot vertical profile of N^2 #####
# Low-pass filter on w_xz
th1_xz_filtered2 = np.zeros_like(th1_xz)

for i in range(NSAMP):
    th1_xz_filtered2[i] = ndimage.gaussian_filter(th1_xz[i],5)

N2_filtered2 = np.gradient(th1_xz_filtered2, gzf, axis=1)
N2t_xz_filtered2 = np.gradient(np.gradient(th1_xz_filtered2, gzf, axis=1), times, axis=0)

fig2, ax2 = plt.subplots(1,3, figsize=(12,6))

xplot = int(3*md['Nx']/8)
a = 10
s = 4

line1, = ax2[0].plot(np.flip(N2t_xz_filtered2[0][:,xplot],axis=0), gzf, color='b')
line2, = ax2[1].plot(np.flip(th1_xz_filtered2[0][:,xplot]-th1_xz_filtered2[0][:,xplot],axis=0), gzf, color='b')
im = ax2[2].imshow(N2t_xz_filtered2[0][:,xplot-a:xplot+a], cmap='bwr', interpolation='bicubic', animated=True,
        extent=[md['LY']*(xplot-a*s)/md['Ny'],md['LY']*(xplot+a*s)/md['Ny'],0,md['LZ']])

fig2.suptitle("time = 0 s")
ax2[0].set_xlabel("x")
ax2[0].set_ylabel("w")
ax2[0].set_xlim(-0.1, 0.1)
ax2[0].set_ylim(0, md['LZ'])

ax2[1].set_xlabel("x")
ax2[1].set_ylabel("b")
ax2[1].set_xlim(-0.01, 0.01)
ax2[1].set_ylim(0, md['LZ'])

im.set_clim(-0.15,0.15)
ax2[2].axhline(md['Lyc']+md['Lyp'],color='black', linestyle=':')
ax2[2].axhline(md['LZ']-md['S_depth'],color='black', linestyle=':')
ax2[2].axvline(xplot*md['LX']/md['Nx'],color='black', linestyle=':')
cb2 = plt.colorbar(im,ax=ax2[2])

def animate2(step):
    line1.set_xdata(np.flip(np.mean(N2t_xz_filtered2[step][:,xplot-5:xplot+5],axis=1),axis=0))
    line2.set_xdata(np.flip(th1_xz_filtered2[step][:,xplot]-th1_xz_filtered2[0][:,xplot],axis=0))
    im.set_data(N2t_xz_filtered2[step][:,xplot-a:xplot+a])
    fig2.suptitle("time = {0:.4f} s".format(step*md['SAVE_STATS_DT']))

    return line1, line2, im,

anim2 = animation.FuncAnimation(fig2, animate2, interval=250, frames=NSAMP)

plt.show()
