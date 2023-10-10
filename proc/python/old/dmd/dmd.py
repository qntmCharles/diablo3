from matplotlib import pyplot as plt
import numpy as np
from pydmd import DMD
from functions import get_metadata, read_params, get_grid

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
    u = np.array([np.array(f['u_xz'][t]) for t in time_keys])
    v = np.array([np.array(f['v_yz'][t]) for t in time_keys])
    NSAMP = len(th1_xz)
    times = np.array([float(f['u_xz'].attrs['Time']) for t in time_keys])
    f.close()

print(times)

