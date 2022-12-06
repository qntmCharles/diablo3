import h5py, gc, sys
import numpy as np
from scipy.interpolate import griddata
from scipy import integrate
import scipy.special as sc
import matplotlib
from datetime import datetime
import matplotlib.animation as animation
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from os.path import join, isfile
from os import listdir
from functions import get_metadata, read_params, get_grid, g2gf_1d

def get_index(z, griddata):
    return int(np.argmin(np.abs(griddata - z)))

def compute_pdf(data, ref, bins, db, normalised=False):
    out_bins = [0 for i in range(len(bins)-1)]

    for i in range(len(bins)-1):
        out_bins[i] = np.sum(np.where(np.logical_and(data > bins[i],
            data < bins[i+1]), ref, 0))

    out_bins = np.array(out_bins)

    if normalised:
        area = integrate.trapezoid(np.abs(out_bins), dx = db)
        return out_bins/area
    else:
        return out_bins

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
    b = g2gf_1d(b)
    t = np.array([np.array(f['th2_xz'][tstep]) for tstep in time_keys])
    t = g2gf_1d(t)
    NSAMP = len(b)
    #times = np.array([float(tstep)*md['SAVE_MOVIE_DT'] for tstep in time_keys])
    times = np.array([float(f['th1_xz'][tstep].attrs['Time']) for tstep in time_keys])
    f.close()

# Compute time indices
nplot = 5
interval = 80

step = np.round(interval*tau / md['SAVE_STATS_DT'])

t_inds = list(map(int,step*np.array(range(1, nplot+1))))
tplot = [times[i] for i in t_inds]
print("Plotting at times: ",tplot)
print("with interval", step*md['SAVE_STATS_DT'])

##### Compute PDF #####
# Create buoyancy bins
b_min = 0
b_max = np.max(b[0,:,int(md['Nx']/2)])/6
nbins = 65

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

##### Compute theoretical PDF #####

rs = np.linspace(0, md['LX']/2, 100)
rm = 1.2 * md['alpha_e'] * md['H']
alpha = md['alpha_e']

theoretical_pdf = (rs/rm)*np.exp(-0.5*np.power(rs/rm,2))

bs = (np.exp(-0.5 * np.power(rs,2) / (6/5 * alpha * md['H'])**2) * 5* F * np.power(
        9/10 * alpha * F,-1/3) * np.power(md['H'],-5/3) / (3*alpha))

#theoretical_pdf /= integrate.trapezoid(theoretical_pdf, bs)

##### Compute reference PDF #####

# First get data we want to process
depth = 5*md['r0']

b_source = b[:,get_index(md['H']-depth, gz):get_index(md['H'], gz)]
t_source = t[:,get_index(md['H']-depth, gz):get_index(md['H'], gz)]

source_pdf = []
for i in range(20,50):
    source_pdf.append(compute_pdf(b_source[i], t_source[i], bbins, db, normalised=True))

source_pdf = np.nanmean(source_pdf, axis=0)

##### Set up plot #####
t_cont = 0.005
X, Y = np.meshgrid(gx, gz[plot_min:plot_max+1])
Xf, Yf = np.meshgrid(gxf, gzf[plot_min:plot_max])
tcols = plt.cm.OrRd(np.linspace(0,1,nplot+1))[1:]

#fig = plt.figure()
fig, ax = plt.subplots(1,2)

ax[0].pcolormesh(X, Y, b[0][plot_min:plot_max], cmap=plt.cm.get_cmap('jet'), alpha=0.3)
ax[1].plot(source_pdf, 0.5*(bbins[1:]+bbins[:-1]), color='k', linestyle='--')
#ax[1].plot(50*theoretical_pdf, bs)

for step,c in zip(t_inds, tcols):
    ax[0].contour(Xf, Yf, t[step][plot_min:plot_max], levels=[t_cont], colors=[c])
    b_pdf = compute_pdf(b[step][plot_min:plot_max], t[step][plot_min:plot_max], bbins, db, normalised=True)

    #plt.plot(source_pdf, 0.5*(bbins[1:]+bbins[:-1]), color=c, linestyle='--')
    ax[1].plot(b_pdf, 0.5*(bbins[1:]+bbins[:-1]), color=c, label = "t={0:.3f} s".format(times[step]))

plt.xlabel("tracer probability density")
plt.ylabel("buoyancy")
plt.legend()
plt.tight_layout()

##### J. Craske (2016) distribution #####
c1 = -25.79
c2 = 1.00
Pe = 10.99
lstar = 1.10

lambdas = np.linspace(0, 4, 1000)

c3 = -np.power(Pe/(2*lstar), 0.5-Pe/2) * sc.gamma(0.5+Pe/2)
lambda_inf = np.linspace(0, 120, 100000)[1:]
c1 = 1/(integrate.trapezoid(np.exp(-Pe*lambda_inf/(2*lstar))*sc.hyp1f1(1, Pe/2+0.5, Pe*lambda_inf/(2*lstar))*\
        np.power(lambda_inf, Pe/2-0.5)+c3,lambda_inf))
c2 = c3*c1

phi = c1 * np.exp(-Pe * lambdas / (2*lstar)) * np.power(lambdas, Pe/2) * sc.hyp1f1(1, Pe/2 + 0.5,\
        Pe*lambdas/(2*lstar)) + c2*np.power(lambdas, 1/2)

print(c1, c2)


phifig = plt.figure()
plt.plot(phi, lambdas)

tfig = plt.figure()
plt.plot(source_pdf, 0.5*(bbins[1:]+bbins[:-1]))
plt.plot(40*phi, lambdas/25)
plt.show()
