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
from functions import get_metadata, read_params, get_grid

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
    NSAMP = len(b)
    #times = np.array([float(tstep)*md['SAVE_MOVIE_DT'] for tstep in time_keys])
    times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

# b contains values of buoyancy at each point
# z_coords contains z value at each point
# t contains tracer value at each point
## Aim is to bin tracer values wrt buoyancy

# Create buoyancy bins
b_min = 0
b_max = np.max(b[0,:,int(md['Nx']/2)])
nbins = 513

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

X, Y = np.meshgrid(gx, gz[plot_min:plot_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[plot_min:plot_max])

contour_isopycnals = np.linspace(b_min,b_max, 20)[1:]
print("Isopycnal contours: ")
print(contour_isopycnals)

fig, ax = plt.subplots(figsize=(15,5))

tplot = ax.pcolormesh(X, Y, t[0][plot_min:plot_max], cmap=plt.cm.get_cmap('OrRd'))

div = make_axes_locatable(ax)

t_cbar_ax = div.append_axes("left", size="5%", pad=0.85)
t_cbar = plt.colorbar(tplot, cax=t_cbar_ax, label="tracer", ticklocation="left")

b_ax = div.append_axes("right", size="100%", pad=0.15, sharey=ax)
bplot = b_ax.pcolormesh(X, Y, b[0][plot_min:plot_max], cmap=plt.cm.get_cmap('jet'))

b_cbar_ax = div.append_axes("right", size="5%", pad=0.65)
b_cbar = plt.colorbar(bplot, cax=b_cbar_ax, label="buoyancy", ticklocation="left")

pdf_ax = div.append_axes("right", size="100%", pad = 0.15, sharey = b_cbar_ax)
pdf, = pdf_ax.plot(np.array(bins)/area, bbins, color='b')

c_isopycnal = b_ax.contour(Xf, Yf, b[0][plot_min:plot_max],
        levels=contour_isopycnals, colors='white', alpha=0.5)
ct_isopycnal = ax.contour(Xf, Yf, b[0][plot_min:plot_max],
        levels=contour_isopycnals, colors='grey', alpha=0.3)

# Labels, decoration etc
bplot.set_clim(b_min, b_max)
tplot.set_clim(0, 0.05)
pdf_ax.set_ylim(b_min, b_max)
pdf_ax.set_xlim(-0.3, 15)

ax.set_ylabel("z")
ax.set_xlabel("x")
b_ax.set_xlabel("x")
pdf_ax.set_xlabel("tracer probability density")

plt.setp(b_ax.get_yticklabels(), visible=False)
plt.setp(pdf_ax.get_yticklabels(), visible=False)

plt.suptitle("time = 0.00 secs")

def animate(step):
    global c_isopycnal
    global ct_isopycnal
    bins = [0 for i in range(nbins)]

    for i in range(nbins-1):
        bins[i] = np.sum(np.where(np.logical_and(b[step][plot_min:plot_max] > bbins[i],
            b[step][plot_min:plot_max] < bbins[i+1]), t[step][plot_min:plot_max], 0))

    area = integrate.trapezoid(np.abs(bins), dx = db)

    bplot.set_array(b[step][plot_min:plot_max].ravel())
    tplot.set_array(t[step][plot_min:plot_max].ravel())
    pdf.set_xdata(np.array(bins)/area)

    for coll in c_isopycnal.collections:
        coll.remove()
    for coll in ct_isopycnal.collections:
        coll.remove()
    c_isopycnal = b_ax.contour(Xf, Yf, b[step][plot_min:plot_max],
            levels=contour_isopycnals, colors='white', alpha=0.5)
    ct_isopycnal = ax.contour(Xf, Yf, b[step][plot_min:plot_max],
            levels=contour_isopycnals, colors='grey', alpha=0.3)

    plt.suptitle("time = {0:.2f} secs".format(times[step]))

    return bplot, tplot, pdf,

anim = animation.FuncAnimation(fig, animate, interval=200, frames=NSAMP)


plt.tight_layout()

if save:
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=5, bitrate=1800)
    now = datetime.now()
    anim.save(save_dir+'tracer%s.mp4'%now.strftime("%d-%m-%Y-%H"),writer=writer)
else:
    plt.show()