# Script loads in 2D slices and produces a movie of the simulation output import numpy as np import h5py, gc
import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid, g2gf_1d

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"

##### ---------------------- #####

def get_index(z, griddata):
    return int(np.floor(np.argmin(np.abs(griddata - z))))

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
    th2_xz = np.array([np.array(f['th2_xz'][t]) for t in time_keys])
    w_xz = np.array([np.array(f['w_xz'][t]) for t in time_keys])

    th1_xz = g2gf_1d(md, th1_xz)
    th2_xz = g2gf_1d(md, th2_xz)
    w_xz = g2gf_1d(md, w_xz)

    NSAMP = len(th1_xz)
    times = np.array([float(t)*md['SAVE_MOVIE_DT'] for t in time_keys])
    f.close()

print("Total time steps: %s"%NSAMP)
print("Dimensional times: ",times)

print("Setting up data arrays...")
fig, axs = plt.subplots(1,2,figsize=(15, 5))
ims = np.array([None,None])
cb = np.array([None,None])

contour_lvls_b = np.linspace(0.01, np.max(th1_xz[0]), 30)
#contour_lvls_trace = np.linspace(0.01, 0.1, 8)
contour_lvls_trace = np.linspace(0.01, 0.2, 8)

print("Setting up initial plot...")
ims[0] = axs[0].pcolormesh(X,Y,th1_xz[0], cmap='jet')
ims[1] = axs[1].pcolormesh(X,Y,th2_xz[0], cmap='jet')
c_b = axs[0].contour(Xf, Yf, th1_xz[0], levels=contour_lvls_b, colors='white', alpha = 0.5)
c_trace = axs[1].contour(Xf, Yf, th2_xz[0], levels=contour_lvls_trace, colors='white', alpha = 0.5)

# Add forcing level
axs[0].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')
axs[1].axhline(md['Lyc']+md['Lyp'],color='white', linestyle=':')

cb[0] = plt.colorbar(ims[0],ax=axs[0])
cb[1] = plt.colorbar(ims[1],ax=axs[1])
ims[0].set_clim(0, np.max(th1_xz[0]))
ims[1].set_clim(0, np.max(th2_xz[-1]))

fig.suptitle("$\\theta_1$, time = 0 hours")
axs[0].set_ylabel("$z$")
axs[1].set_ylabel("$z$")
axs[0].set_xlabel("$x$")
axs[1].set_xlabel("$y$")

def animate(step):
    global c_b, c_trace

    ims[0].set_array(th1_xz[step].ravel())
    ims[1].set_array(th2_xz[step].ravel())
    fig.suptitle("$\\theta_1$, time = {0:.4f} hours".format(times[step]/3600))

    for coll in c_b.collections:
        coll.remove()
    for coll in c_trace.collections:
        coll.remove()

    c_b = axs[0].contour(Xf, Yf, th1_xz[step], levels=contour_lvls_b, colors='white', alpha = 0.5)
    c_trace = axs[1].contour(Xf, Yf, th2_xz[step], levels=contour_lvls_trace, colors='white', alpha = 0.5)

    return ims.flatten(),

print("Initialising mp4 writer...")
Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, bitrate=1800)

print("Starting plot...")
anim = animation.FuncAnimation(fig, animate, interval=20, frames=NSAMP)
now = datetime.now()
#anim.save(save_dir+'jet_%s.mp4'%now.strftime("%d-%m-%Y-%H-%m"),writer=writer)

#############################################################################################################

alpha = md['alpha_e']
zvirt = -5/6 * md['r0']/alpha
F0 = md['r0']**2 * md['b0']

r_m = np.zeros(shape=(md['Nz'],))
b_m = np.zeros(shape=(md['Nz'],))
w_m = np.zeros(shape=(md['Nz'],))

b_forcing = np.zeros(shape=(md['Nz'],md['Nx']+1))
w_forcing = np.zeros(shape=(md['Nz'],md['Nx']+1))

for j in range(md['Nz']):
    r_m[j] = 1.2 * alpha * (gzf[j] - zvirt)
    w_m[j] = (0.9 * alpha * F0)**(1/3) * (gzf[j]-zvirt)**(2/3)/r_m[j]
    b_m[j] = F0/(r_m[j] * r_m[j] * w_m[j])

    w_forcing[j] = 0.2 * w_m[j] * (np.tanh((gx - md['LX']/2+4*r_m[j])/1e-3) \
                            -np.tanh((gx - md['LX']/2-4*r_m[j])/1e-3)) \
                            *(1 - np.tanh((gzf[j]-md['Lyc'])/md['Lyp']))
    b_forcing[j] = 0.2 * b_m[j] * (np.tanh((gx - md['LX']/2+4*r_m[j])/1e-3) \
                            -np.tanh((gx - md['LX']/2-4*r_m[j])/1e-3)) \
                            *(1 - np.tanh((gzf[j]-md['Lyc'])/md['Lyp']))

strat_idx = get_index(0.8*md['H'], gzf)

skip = 2
cols = plt.cm.rainbow(np.linspace(0,1,int(get_index(md['Lyc'],gzf)/skip)))

for j in [0, 1, 2, 4, 7]:
    cs_fig = plt.figure()
    plt.title("t = %s"%str(times[j]))
    for i,c in zip(range(0,get_index(md['Lyc'],gzf),skip),cols):
        plt.plot(gxf, th2_xz[j, i, :], color=c)
        plt.plot(gx, b_forcing[i], color=c, linestyle='--')

    plt.xlim(0.2,0.4)
    plt.ylim(0, 0.3)
    #plt.show()

summary_fig, ax = plt.subplots(1,2)

tstart_idx = get_index(3, times)
cols = plt.cm.rainbow(np.linspace(0,1,NSAMP-tstart_idx))

for i, c in zip(range(tstart_idx,NSAMP),cols):
    ax[1].plot(gxf, th2_xz[i, strat_idx, :], color=c, alpha=0.3)

ax[1].plot(gxf, np.mean(th2_xz[tstart_idx:, strat_idx, :], axis=0), color='k', linestyle='--')

for i, c in zip(range(2,10),cols):
    ax[0].plot(gxf, th2_xz[i, 5, :], color=c, alpha=0.3)

ax[0].plot(gxf, np.mean(th2_xz[2:10, 5, :], axis=0), color='k', linestyle='--')
ax[0].plot(gx, b_forcing[5], color='k', linestyle='--')

ax[0].set_xlim(0.2, 0.4)
ax[1].set_xlim(0.2, 0.4)

plt.show()
