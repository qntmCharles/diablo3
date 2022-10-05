import h5py, bisect, time, gc, sys
from os import listdir
from os.path import isfile, join
import numpy as np
from math import sqrt
import matplotlib.patheffects as pe
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from functions import get_metadata, get_grid, read_params, get_az_data
from scipy import integrate, optimize, interpolate
from matplotlib import cm as cm
from itertools import groupby
from mpl_toolkits.axes_grid1 import make_axes_locatable

##### USER-DEFINED PARAMETERS #####
params_file = "./params.dat"
save_loc = "/home/cwp29/Documents/posters/issf2/"
save = False
title = False
show = True

blue = (19/256, 72/256, 158/256)

z_upper = 70 # non-dim, scaled by r_0
z_lower = 35

eps = 0.02

matplotlib.rcParams.update({'axes.labelsize': 'large'})

##### ----------------------- #####

def ranges(nums):
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    return list(zip(edges, edges))

def truncate(var, points, ref_var, ref, trunc_indices, display=False):
    # Truncates var at location where ref_var matches ref, computing only at trunc_indices
    # ASSUMES ref_var is decreasing with increasing index
    # Points are the positions of each entry of var
    var_new = np.zeros(shape=(var.shape[0], var.shape[1]+2))
    for i in trunc_indices:
        # Calculate index up to which var is unchanged, after which set to 0
        trunc_idx = ranges(np.where(ref_var[i,:] > ref[i])[0])[0][-1]
        var_new[i, :trunc_idx+1] = var[i, :trunc_idx+1]

        # Calculate exact point to interpolate
        ref_var_interp = ref_var[i, trunc_idx:trunc_idx+2]
        points_interp = points[trunc_idx:trunc_idx+2]
        f = interpolate.interp1d(ref_var_interp, points_interp)
        trunc_pt = f(ref[i])

        # Create interpolation function for var
        points_interp = points[trunc_idx:trunc_idx+2]
        var_interp = var[i,trunc_idx:trunc_idx+2]

        f = interpolate.interp1d(points_interp, var_interp)
        var_new[i, trunc_idx+1] = f(trunc_pt)

    return var_new


##### Get directory locations #####
base_dir, run_dir, save_dir, version = read_params(params_file)
print("Run directory: ", run_dir)
print("Save directory: ", save_dir)

##### Get simulation metadata #####
md = get_metadata(run_dir, version)
print("Complete metadata: ",md)

##### Get grid #####
gxf, gyf, gzf, dz = get_grid(join(run_dir, 'grid.h5'), md)
gzfp = np.flip(gzf)

r_0 = md['r0']
dr = md['LX']/md['Nx']
nbins = int(md['Nx']/2)
r_bins = np.array([r*dr for r in range(0, nbins+1)])
r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])

##### Get azimuthal data #####
############################ TODO: check this works okay with very large data files
data = get_az_data(join(save_dir,'az_stats.h5'), md)

ubar = data['u']
vbar = data['v']
wbar = data['w']
bbar = data['b']
pbar = data['p']

ufluc2bar = data['ufluc2']
uflucvflucbar = data['uflucvfluc']
uflucwflucbar = data['uflucwfluc']
uflucbflucbar = data['uflucbfluc']
vfluc2bar = data['vfluc2']
vflucwflucbar = data['vflucwfluc']
wfluc2bar = data['wfluc2']
wflucbflucbar = data['wflucbfluc']
bfluc2bar = data['bfluc2']

dwbardz = np.gradient(wbar, gzf, axis=0)
dwbardr = np.gradient(wbar, dr, axis=1)

dpbardz = np.gradient(pbar, gzf, axis=0)
dpbardr = np.gradient(pbar, dr, axis=1)

##### Identify indices where plume is well defined #####

# find indices where centreline velocity is positive
valid_indices = [j for j in range(wbar.shape[0]) if wbar[j,0] > 0]
cont_valid_indices = list(set(list(sum(max([list(y) for i, y in groupby(zip(valid_indices,
    valid_indices[1:]), key = lambda x: (x[1]-x[0]) == 1)], key=len),()))))
if cont_valid_indices[-1] == md['Nz']-1:
    cont_valid_indices.remove(md['Nz']-1)

# check where rtrunc is, if at the edge of domain then remove index from list
plume_indices = []
z_upper_ind = int(max(np.where(gzf <= z_upper*r_0)[0]))
z_lower_ind = int(min(np.where(z_lower*r_0 <= gzf)[0]))
for j in cont_valid_indices:
    wtrunc = 0.02*wbar[j, 0]
    rtrunc = max(np.where(wbar[j,:] > wtrunc)[0])
    if (z_lower_ind <= j <= z_upper_ind) and rtrunc != md['Nx']/2-1:
        plume_indices.append(j)

# This array contains the largest continuous run of indices which have a positive centreline velocity
# and valid truncation radius, at z levels between z_lower and z_upper
plume_indices = list(set(list(sum(max([list(y) for i, y in groupby(zip(plume_indices,
    plume_indices[1:]), key = lambda x: (x[1]-x[0]) == 1)], key=len),()))))

print(plume_indices)

##### Truncate data at radius where wbar(r) = eps*wbar(0) #####

Q_full = 2*integrate.trapezoid(wbar*r_points, r_points, axis=1)
M_full = 2*integrate.trapezoid(wbar*wbar*r_points, r_points, axis=1)
F_full = 2*integrate.trapezoid(wbar*bbar*r_points, r_points, axis=1)
B_full = 2*integrate.trapezoid(bbar*r_points, r_points, axis=1)

r_d = []
wbar_trunc = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
r_integrate = np.zeros(shape=(wbar.shape[0],wbar.shape[1]+2))
for i in cont_valid_indices:
    wtrunc = eps*wbar[i,0]
    try:
        rtrunc = ranges(np.where(wbar[i,:] > wtrunc)[0])[0][-1]
    except ValueError:
        cont_valid_indices.remove(i)
        continue

    wbar_trunc[i, :rtrunc+1] = wbar[i, :rtrunc+1]
    wbar_trunc[i, rtrunc+1] = wtrunc

    # Edge cases
    if rtrunc == int(md['Nx']/2)-1:
        wtrunc = wbar[i,-1]
    if rtrunc == 0:
        wtrunc = wbar[i, 0]

    wbar_interp = wbar[i, rtrunc:rtrunc+2]
    r_interp = r_points[rtrunc:rtrunc+2]

    f = interpolate.interp1d(wbar_interp, r_interp)
    r_d.append(f(wtrunc))
    r_integrate[i, :rtrunc+1] = r_points[:rtrunc+1]
    r_integrate[i, rtrunc+1] = f(wtrunc)
    r_integrate[i, rtrunc+2] = f(wtrunc)

bbar_trunc = truncate(bbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
pbar_trunc = truncate(pbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
wfluc2bar_trunc = truncate(wfluc2bar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
ufluc2bar_trunc = truncate(ufluc2bar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
vfluc2bar_trunc = truncate(vfluc2bar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
uflucwflucbar_trunc = truncate(uflucwflucbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
wflucbflucbar_trunc = truncate(wflucbflucbar, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
dwbardz_trunc = truncate(dwbardz, r_points, wbar, eps*wbar[:,0], cont_valid_indices)
dwbardr_trunc = truncate(dwbardr, r_points, wbar, eps*wbar[:,0], cont_valid_indices)

##### Calculate integral quantities and profile coefficients #####

Q = 2*integrate.trapezoid(wbar_trunc*r_integrate, r_integrate, axis=1)
M = 2*integrate.trapezoid(wbar_trunc*wbar_trunc*r_integrate, r_integrate, axis=1)
F = 2*integrate.trapezoid(wbar_trunc*bbar_trunc*r_integrate, r_integrate, axis=1)
B = 2*integrate.trapezoid(bbar_trunc*r_integrate, r_integrate, axis=1)
print("Q:", Q)
print("M:", M)
print("F:", F)

r_m = Q/np.sqrt(M)
w_m = M/Q
b_m = B/(r_m*r_m)

''' PLOTTING '''
rs = np.linspace(0, 2, 1000)
analytic_profile = 2*np.exp(-2*np.power(rs,2))
analytic_profile2 = 8*0.025*rs*np.exp(-2*np.power(rs+0.05,2))
analytic_profile3 = 8*0.045*rs*np.exp(-2*np.power(rs+0.05,2))

#fig2, axs2 = plt.subplots(1,2, figsize=(8,4), facecolor=(0.9,0.9,0.9))
fig2, axs2 = plt.subplots(1,2, figsize=(8,4))
if title: fig2.suptitle("Fig 4(c),(f): self-similar profiles of $\overline{w}, \overline{b}, \overline{u'w'}, \overline{u'b'}$")
for i in plume_indices:
    if i == plume_indices[0]:
        axs2[0].plot(r_points/r_m[i], wbar[i]/w_m[i],color='blue',label="$\overline{w}/w_m$")
        axs2[0].plot(r_points/r_m[i], bbar[i]/b_m[i],color='red',linestyle='dashed',label="$\overline{b}/b_m$")
        axs2[1].plot(r_points/r_m[i], uflucwflucbar[i]/(w_m[i]*w_m[i]), color='blue',
                label="$\overline{u'w'}/w_m^2$")
        axs2[1].plot(r_points/r_m[i], uflucbflucbar[i]/(w_m[i]*b_m[i]), color='red',
                linestyle='dashed', label="$\overline{u'b'}/w_m b_m$")
    else:
        axs2[0].plot(r_points/r_m[i], wbar[i]/w_m[i],color='blue')
        axs2[0].plot(r_points/r_m[i], bbar[i]/b_m[i],color='red',linestyle='dashed')
        axs2[1].plot(r_points/r_m[i], uflucwflucbar[i]/(w_m[i]*w_m[i]), color='blue')
        axs2[1].plot(r_points/r_m[i], uflucbflucbar[i]/(w_m[i]*b_m[i]), color='red', linestyle='dashed')

#axs2[0].plot(rs, analytic_profile, color='white', linestyle='--', linewidth=2.0, label="analytic")
axs2[0].plot(rs, analytic_profile, color='white', path_effects = [pe.Stroke(linewidth=3,
    foreground='k'), pe.Normal()], label="analytic")
axs2[1].plot(rs, analytic_profile2, color='white', path_effects = [pe.Stroke(linewidth=3,
    foreground='k'), pe.Normal()], label="analytic")
axs2[1].plot(rs, analytic_profile3, color='white', path_effects = [pe.Stroke(linewidth=3,
    foreground='k'), pe.Normal()])
axs2[0].set_xlim(0, 2)
axs2[0].set_xlabel("$r/r_m$")
#l1 = axs2[0].legend(facecolor=(0.9,0.9,0.9))
l1 = axs2[0].legend()
axs2[1].set_xlim(0,2)
axs2[1].set_xlabel("$r/r_m$")
#l2 = axs2[1].legend(facecolor=(0.9,0.9,0.9))
l2 = axs2[1].legend()

#for text in l1.get_texts():
    #text.set_color(blue)
#for text in l2.get_texts():
    #text.set_color(blue)

nticks = 5
ticks = np.linspace(0,2,nticks)
axs2[1].set_xticks(ticks)
axs2[1].set_xticklabels(["{0:.1f}".format(i) for i in ticks])
axs2[0].set_xticks(ticks)
axs2[0].set_xticklabels(["{0:.1f}".format(i) for i in ticks])
#axs2[0].grid(color='grey', alpha=0.5)
#axs2[1].grid(color='grey', alpha=0.5)

#axs2[0].set_facecolor((0.9,0.9,0.9))
#axs2[1].set_facecolor((0.9,0.9,0.9))
#axs2[0].tick_params(color=blue, labelcolor=blue)
#axs2[1].tick_params(color=blue, labelcolor=blue)

#for spine in axs2[0].spines.values():
    #spine.set_edgecolor(blue)
#for spine in axs2[1].spines.values():
    #spine.set_edgecolor(blue)

fig2.tight_layout()
if save: fig2.savefig(save_loc+'fig2.png', dpi=300)
fig2.savefig(save_loc+'wb_uw_ub.png', dpi=300)
plt.show()
