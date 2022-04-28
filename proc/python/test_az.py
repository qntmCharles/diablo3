import h5py, bisect, time, gc, sys
from os import listdir
from os.path import isfile, join
import numpy as np
from math import sqrt, floor
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from functions import get_metadata, get_grid, read_params, get_az_data
from scipy import integrate, optimize, interpolate
from matplotlib import cm as cm
from itertools import groupby

##### USER-DEFINED PARAMETERS #####
params_file = "./params.dat"

out_file = "out.000095.h5"

##### ----------------------- #####

def horizontal_interpolate(var, md):
    var_interp = np.zeros_like(var)
    for k in range(md['Nz']):
        for i in range(md['Nx']):
            for j in range(md['Ny']):
                # edge cases
                if (i == md['Nx'] - 1) and (j == md['Ny'] - 1):
                    var_interp[i,j,k] = 0.25 * (var[i,j,k] + var[0,j,k] + var[i,0,k] + var[0,0,k])
                elif i == md['Nx'] - 1:
                    var_interp[i,j,k] = 0.25 * (var[i,j,k] + var[0,j,k] + var[i,j+1,k] + var[0,j+1,k])
                elif j == md['Ny'] - 1:
                    var_interp[i,j,k] = 0.25 * (var[i,j,k] + var[i+1,j,k] + var[i,0,k] + var[i+1,0,k])
                else: # main case
                    var_interp[i,j,k] = 0.25 * (var[i,j,k] + var[i+1,j,k] + var[i,j+1,k] + var[i+1,j+1,k])

    return var_interp



##### ----------------------- #####

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

x_coords, y_coords, z_coords = np.meshgrid(gxf, gyf, gzf, indexing='ij', sparse=True)
radii = np.sqrt((x_coords-md['LX']/2)**2 + (y_coords-md['LY']/2)**2)
radii_m = np.where(radii <= md['LX']/2, radii, 0)
radii_n = np.where(radii <= md['LX']/2, radii, np.nan)

r_0 = md['r0']
dr = md['LX']/md['Nx']
nbins = int(md['Nx']/2)
r_bins = np.array([r*dr for r in range(0, nbins+1)])
r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])
r_inds = np.searchsorted(r_bins[1:], radii_m)[:,:,0]
r_inds_full = np.searchsorted(r_bins[1:], radii_n)

#calculate number of grid cells in each radius bin
r_inds_n = np.searchsorted(r_bins[1:], radii_n)[:,:,0]
n_shells = [np.sum(np.where(r_inds_n == i, 1, 0)) for i in range(nbins)]


##### Load full 3D data #####
with h5py.File(join(save_dir, out_file),'r') as f:
    print(f['Timestep'].keys())
    u = np.array(f['Timestep']['U'])
    v = np.array(f['Timestep']['V'])
    w = np.array(f['Timestep']['W'])
    b = np.array(f['Timestep']['TH1'])
    p = np.array(f['Timestep']['P'])
    u_r = np.array(f['Timestep']['UR'])
    u_theta = np.array(f['Timestep']['UTHETA'])

    u = np.transpose(u,axes=(2,0,1))
    v = np.transpose(v,axes=(2,0,1))
    w = np.transpose(w,axes=(2,0,1))
    b = np.transpose(b,axes=(2,0,1)) # is this correct???
    p = np.transpose(p,axes=(2,0,1))
    u_r = np.transpose(u_r,axes=(2,0,1))
    u_theta = np.transpose(u_theta,axes=(2,0,1))

    time = f['Timestep'].attrs['Time']
    print("3D full data arrays have shape ",u.shape)
    #u, v, w, b are on the FRACTIONAL vertical grid, so no need to interpolate in vertical
    # but we DO need to interpolate in the HORIZONTAL
    u = horizontal_interpolate(u,md)
    v = horizontal_interpolate(v,md)
    w = horizontal_interpolate(w,md)
    b = horizontal_interpolate(b,md)
    p = horizontal_interpolate(p,md)

    # Compute radial and azimuthal velocity
    radii = np.sqrt((x_coords-md['LX']/2)**2 + (y_coords-md['LX']/2)**2)
    ur = ((x_coords-md['LX']/2)*u + (y_coords-md['LX']/2)*v)/radii
    utheta = ((x_coords-md['LX']/2)*v - (y_coords-md['LX']/2)*u)/radii

##### Get azimuthal data #####
with h5py.File(join(save_dir, 'az_stats.h5'), 'r') as f:
    ndata = len(f['u_az'].keys())
    tkeys = list(f['u_az'].keys())
    for i in range(ndata):
        test_time = f['u_az'][tkeys[i]].attrs['Time']
        if test_time == time:
            u_az = np.array(f['u_az'][tkeys[i]])
            v_az = np.array(f['v_az'][tkeys[i]])
            w_az = np.array(f['w_az'][tkeys[i]])
            b_az = np.array(f['b_az'][tkeys[i]])
            p_az = np.array(f['p_az'][tkeys[i]])

print("2D azimuthal data arrays have shape ",u_az.shape)

counts = np.zeros(shape=(md['Nz'], nbins))
u_binned = np.zeros(shape=(md['Nz'], nbins))
v_binned = np.zeros(shape=(md['Nz'], nbins))
w_binned = np.zeros(shape=(md['Nz'], nbins))
b_binned = np.zeros(shape=(md['Nz'], nbins))
p_binned = np.zeros(shape=(md['Nz'], nbins))

for i in range(md['Nx']):
    for j in range(md['Ny']):
        for k in range(md['Nz']):
            bin_idx = floor(sqrt((gxf[i]-md['LX']/2)**2 + (gyf[j]-md['LY']/2)**2)*md['Nx']/md['LX'])
            if bin_idx < nbins:
                counts[k, bin_idx] += 1
                u_binned[k, bin_idx] += ur[i,j,k]
                v_binned[k, bin_idx] += utheta[i,j,k]
                w_binned[k, bin_idx] += w[i,j,k]
                b_binned[k, bin_idx] += b[i,j,k]
                p_binned[k, bin_idx] += p[i,j,k]


u_comp = u_binned/counts
v_comp = v_binned/counts
w_comp = w_binned/counts
b_comp = b_binned/counts
p_comp = p_binned/counts

##### Read output file and extract data #####
bs = []
us = []
vs = []
ws = []
ps = []
uu_sflucs = []

b_from_output = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
w_from_output = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
u_from_output = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
v_from_output = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
p_from_output = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
us_from_output = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))

with open(join(save_dir, 'output.dat'), 'r') as f:
    text = f.readlines()
    cur_time_step = 0
    for i in range(len(text)):
        strings = text[i].split()
        if len(strings) > 0 and strings[0] == "DATA":
            strings.extend(text[i+1].split())
            #print(strings)
            time_step = int(strings[2])
            if time_step != cur_time_step:
                bs.append(b_from_output)
                ws.append(w_from_output)
                us.append(u_from_output)
                vs.append(v_from_output)
                ps.append(p_from_output)
                uu_sflucs.append(us_from_output)
                b_from_output = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
                w_from_output = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
                u_from_output = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
                v_from_output = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
                us_from_output = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
                cur_time_step = time_step

            gname = strings[1]
            i = int(strings[3])
            j = int(strings[4])
            k = int(strings[5])-1
            data = float(strings[6])

            if gname == "w_az":
                w_from_output[i,j,k] = data
            elif gname == "b_az":
                b_from_output[i,j,k] = data
            elif gname == "u_az":
                u_from_output[i,j,k] = data
            elif gname == "v_az":
                v_from_output[i,j,k] = data
            elif gname == "p_az":
                p_from_output[i,j,k] = data
            elif gname == "uu_sfluc":
                us_from_output[i,j,k] = data
            else:
                print("bork")
    bs.append(b_from_output)
    ws.append(w_from_output)
    us.append(u_from_output)
    vs.append(v_from_output)
    ps.append(p_from_output)
    uu_sflucs.append(us_from_output)

TSTEP = 3

uu_sfluc = np.zeros(shape=(md['Nx'], md['Ny'], md['Nz']))
for i in range(md['Nx']):
    for j in range(md['Ny']):
        for k in range(md['Nz']):
            bin_idx = floor(sqrt((gxf[i]-md['LX']/2)**2 + (gyf[j]-md['LY']/2)**2)*md['Nx']/md['LX'])
            if bin_idx < nbins:
                uu_sfluc[i,j,k] = (us[TSTEP][i,j,k] - u_az[k, bin_idx])**2

NrankZ = 4
NrankY = 8
RANKZ = 3
RANKY = 0
Nzp = int((md['Nz']-1)/NrankZ)
Nyp = int(md['Ny']/NrankY)

jmin = Nyp*RANKZ
jmax = Nyp*(RANKZ+1)
print(jmin,jmax)

fig, ax = plt.subplots(1,2)
ax[0].imshow(uu_sfluc[:,jmin:jmax,8])
ax[1].imshow(uu_sflucs[TSTEP][:,jmin:jmax,8])
plt.show()

bin_idxs = np.floor(np.sqrt((x_coords-md['LX']/2)**2 + (y_coords-md['LY']/2)**2) * md['Nx']/md['LX'])
bin_idxs = bin_idxs.astype(np.int64)
bin_idxs = np.where(bin_idxs < nbins, bin_idxs, 0)
print(bin_idxs)
