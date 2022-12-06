import h5py
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
from functions import get_metadata, read_params, get_grid

##### USER DEFINED VARIABLES #####

params_file = "./params.dat"
var1 = 'b_az'
var2 = 'uw_sfluc'
var3 = 'ub_sfluc'
save = False

##### ---------------------- #####

# Get dir locations from param file
base_dir, run_dir, save_dir, version = read_params(params_file)
print(run_dir)
print(save_dir)

# Get simulation metadata
md = get_metadata(run_dir, version)

gxf, gyf, gzf, dz = get_grid(save_dir+'grid.h5', md)

print("Complete metadata: ", md)

plt.plot(gzf)
plt.show()
