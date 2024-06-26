import h5py
import numpy as np
from matplotlib import pyplot as plt
from os.path import join
from scipy import integrate
import fortranformat as ff

def get_index(z, griddata):
    return int(np.floor(np.argmin(np.abs(griddata - z))))

def get_plotindex(z, frac_grid):
    return int(frac_grid.searchsorted(z, 'left'))

def read_params(param_file):
    params = [i.strip() for i in open(param_file)]

    for line in params:
        dir_type = line.split("=")[0]
        dir_location = line.split("=")[1]
        if dir_type == "base_dir":
            base_dir = dir_location
        if dir_type == "save_dir":
            save_dir = base_dir + dir_location
        if dir_type == "run_dir":
            run_dir = base_dir + dir_location
        if dir_type == "current_version":
            current_version = dir_location

    return base_dir, run_dir, save_dir, current_version

def get_metadata(run_dir, version):
    md = {}

    #if version == "3.5":
        #params = ["LX", "LY", "LZ", "RE", "SAVE_MOVIE_DT", "SAVE_STATS_DT", "SAVE_FLOW_DT"]
        #chan_params = ["r0", "alpha_e", "Q0", "Lyc", "Lyp"]
        #grid_params = ["Nx", "Ny", "Nz", "Nth"]
    #if version == "3.6":
        #params = ["LX", "LY", "LZ", "RE", "SAVE_MOVIE_DT", "SAVE_STATS_DT", "SAVE_FLOW_DT"]
        #chan_params = ["r0", "alpha_e", "b0", "Lyc", "Lyp"]
        #grid_params = ["Nx", "Ny", "Nz", "Nth"]
    #if version == "3.7":
    params = ["LX", "LY", "LZ", "RE", "SAVE_MOVIE_DT", "SAVE_STATS_DT", "SAVE_FLOW_DT", "VERSION"]
    chan_params = ["r0", "alpha_e", "b0", "Lyc", "Lyp", "S_depth", "N2", "H", "Q0"]
    grid_params = ["Nx", "Ny", "Nz", "Nth"]
    if version == "3.8":
        chan_params = ["r0", "alpha_e", "b0", "Lyc", "Lyp", "S_depth", "N2", "H",
                "Q0","b_factor", "phi_factor", "Nb", "Nphi","F_TYPE", "Omega_thresh"]
        params = ["LX", "LY", "LZ", "RE", "SAVE_MOVIE_DT", "SAVE_STATS_DT",
                "LES_DT_END", "NU_START", "NU_RUN", "SAVE_FLOW_DT", "VERSION"]

    for params_file, parameters in zip(["/input.dat", "/input_chan.dat"],[params,chan_params]):
        with open(run_dir+params_file, 'r') as f:
            lines = list(f.read().splitlines())
            f.close()

        for name in parameters:
            param_line = -1
            param_col = 1
            for i in range(len(lines)):
                words = [y.split(',')[0] for y in lines[i].split()]
                if name in words:
                    param_line = i
                    if name in ["SAVE_MOVIE_DT", "SAVE_FLOW_DT", "SAVE_STATS_DT"]:
                        param_col = words.index(name)-len(words)-1
                    else:
                        param_col = words.index(name)-len(words)

            if param_line > 0:
                if name in ["NU_RUN", "Omega_thresh"]:
                    reader = ff.FortranRecordReader('ES12.5')
                    md[name] = reader.read(lines[param_line+1].split()[param_col])[0]
                else:
                    md[name] = float(lines[param_line+1].split()[param_col])
                    if name == "RE":
                        md['nu'] = 1/md["RE"]


    with open(run_dir+"/grid_def.all",'r') as f:
        lines = list(f.read().splitlines())
        for name in grid_params:
            for i in range(len(lines)):
                words = lines[i].split()
                if name in words:
                    md[name] = int(words[-1])

    return md

def get_grid(grid_file, md, fractional_grid=True):
    with h5py.File(grid_file, 'r') as f:
        gz = np.array(f['grids']['y'])
        f.close()

    gzf = np.zeros((md['Nz']+2,))
    gzf[1:-1] = 0.5*(gz[:-1]+gz[1:])
    gzf[0] = 2*gzf[1]-gzf[2]
    gzf[-1] = 2*gzf[-2]-gzf[-3]

    dz = np.zeros((md['Nz']+1,))
    dz[:] = gzf[1:]-gzf[:-1]

    gzf = gzf[1:-1]

    gx = np.linspace(0, md['LX'], md['Nx']+1)
    gxf = 0.5*(gx[:-1] + gx[1:])
    gy = np.linspace(0, md['LY'], md['Ny']+1)
    gyf = 0.5*(gy[:-1] + gy[1:])

    if fractional_grid:
        return gxf, gyf, gzf, dz
    else:
        return gx, gy, gz, gz[1:]-gz[:-1]

def get_az_data(data_file, md, tstart_ind=0, verbose=True):
    data_dict = {}
    with h5py.File(data_file,'r') as f:
        if verbose: print("Data keys: %s"%f.keys())

        # Time data
        time_keys = list(f['u_az'].keys())
        time_keys = time_keys[tstart_ind:]

        # (u,v,w,b,p,t) data
        u_az = f['u_az']
        v_az = f['v_az']
        w_az = f['w_az']
        b_az = f['b_az']
        p_az = f['p_az']
        th_az = f['th_az']

        # Spatial fluctuations
        uu_sfluc = f['uu_sfluc']
        uv_sfluc = f['uv_sfluc']
        uw_sfluc = f['uw_sfluc']
        ub_sfluc = f['ub_sfluc']
        vv_sfluc = f['vv_sfluc']
        vw_sfluc = f['vw_sfluc']
        ww_sfluc = f['ww_sfluc']
        wb_sfluc = f['wb_sfluc']
        bb_sfluc = f['bb_sfluc']

        ##### Calculate time averages ######
        if verbose: print("Calculating time averages...")

        # Initialise
        if u_az[time_keys[0]].attrs['Time'] == 0: # if file starts at t=0, skip first dataset
            t_start = 0
            time_keys = time_keys[1:]
        else:
            t_start = u_az[time_keys[0]].attrs['Time']-md['SAVE_STATS_DT']
        t_end = u_az[time_keys[-1]].attrs['Time']
        t_run = t_end - t_start
        if verbose: print(t_run)

        if verbose: print("Computing t = {0:.4f}".format(u_az[time_keys[0]].attrs['Time']))
        ubar = u_az[time_keys[0]][()]*md['SAVE_STATS_DT']
        vbar = v_az[time_keys[0]][()]*md['SAVE_STATS_DT']
        wbar = w_az[time_keys[0]][()]*md['SAVE_STATS_DT']
        bbar = b_az[time_keys[0]][()]*md['SAVE_STATS_DT']
        pbar = p_az[time_keys[0]][()]*md['SAVE_STATS_DT']
        thbar = th_az[time_keys[0]][()]*md['SAVE_STATS_DT']

        uu_sfluc_bar = uu_sfluc[time_keys[0]][()]*md['SAVE_STATS_DT']
        uv_sfluc_bar = uv_sfluc[time_keys[0]][()]*md['SAVE_STATS_DT']
        uw_sfluc_bar = uw_sfluc[time_keys[0]][()]*md['SAVE_STATS_DT']
        ub_sfluc_bar = ub_sfluc[time_keys[0]][()]*md['SAVE_STATS_DT']
        vv_sfluc_bar = vv_sfluc[time_keys[0]][()]*md['SAVE_STATS_DT']
        vw_sfluc_bar = vw_sfluc[time_keys[0]][()]*md['SAVE_STATS_DT']
        ww_sfluc_bar = ww_sfluc[time_keys[0]][()]*md['SAVE_STATS_DT']
        wb_sfluc_bar = wb_sfluc[time_keys[0]][()]*md['SAVE_STATS_DT']
        bb_sfluc_bar = bb_sfluc[time_keys[0]][()]*md['SAVE_STATS_DT']

        # Compute sum
        for i in range(1,len(time_keys)):
            t_key = time_keys[i]
            dt = u_az[t_key].attrs['Time'] - u_az[time_keys[i-1]].attrs['Time']

            if verbose: print("Computing t = {0:.4f}".format(u_az[t_key].attrs['Time']))
            ubar += u_az[t_key][()]*dt
            vbar += v_az[t_key][()]*dt
            wbar += w_az[t_key][()]*dt
            bbar += b_az[t_key][()]*dt
            pbar += p_az[t_key][()]*dt
            thbar += th_az[t_key][()]*dt

            uu_sfluc_bar += uu_sfluc[t_key][()]*dt
            uv_sfluc_bar += uv_sfluc[t_key][()]*dt
            uw_sfluc_bar += uw_sfluc[t_key][()]*dt
            ub_sfluc_bar += ub_sfluc[t_key][()]*dt
            vv_sfluc_bar += vv_sfluc[t_key][()]*dt
            vw_sfluc_bar += vw_sfluc[t_key][()]*dt
            ww_sfluc_bar += ww_sfluc[t_key][()]*dt
            wb_sfluc_bar += wb_sfluc[t_key][()]*dt
            bb_sfluc_bar += bb_sfluc[t_key][()]*dt

        ubar /= t_run
        vbar /= t_run
        wbar /= t_run
        bbar /= t_run
        pbar /= t_run
        thbar /= t_run

        data_dict['u'] = ubar
        data_dict['v'] = vbar
        data_dict['w'] = wbar
        data_dict['p'] = pbar
        data_dict['b'] = bbar
        data_dict['th'] = thbar

        uu_sfluc_bar /= t_run
        uv_sfluc_bar /= t_run
        uw_sfluc_bar /= t_run
        ub_sfluc_bar /= t_run
        vv_sfluc_bar /= t_run
        vw_sfluc_bar /= t_run
        ww_sfluc_bar /= t_run
        wb_sfluc_bar /= t_run
        bb_sfluc_bar /= t_run

        # Initialise
        if verbose: print("Calculating fluctuations...")
        if verbose: print("Computing t = {0:.4f}".format(u_az[time_keys[0]].attrs['Time']))
        u_tfluc = u_az[time_keys[0]][()] - ubar
        v_tfluc = v_az[time_keys[0]][()] - vbar
        w_tfluc = w_az[time_keys[0]][()] - wbar
        b_tfluc = b_az[time_keys[0]][()] - bbar

        uu_tfluc_bar = u_tfluc*u_tfluc*md['SAVE_STATS_DT']
        uv_tfluc_bar = u_tfluc*v_tfluc*md['SAVE_STATS_DT']
        uw_tfluc_bar = u_tfluc*w_tfluc*md['SAVE_STATS_DT']
        ub_tfluc_bar = u_tfluc*b_tfluc*md['SAVE_STATS_DT']
        vv_tfluc_bar = v_tfluc*v_tfluc*md['SAVE_STATS_DT']
        vw_tfluc_bar = v_tfluc*w_tfluc*md['SAVE_STATS_DT']
        ww_tfluc_bar = w_tfluc*w_tfluc*md['SAVE_STATS_DT']
        wb_tfluc_bar = w_tfluc*b_tfluc*md['SAVE_STATS_DT']
        bb_tfluc_bar = b_tfluc*b_tfluc*md['SAVE_STATS_DT']

        # Compute sum
        for i in range(1,len(time_keys)):
            t_key = time_keys[i]
            dt = u_az[t_key].attrs['Time'] - u_az[time_keys[i-1]].attrs['Time']

            if verbose: print("Computing t = {0:.4f}".format(u_az[t_key].attrs['Time']))
            u_tfluc = u_az[t_key][()] - ubar
            v_tfluc = v_az[t_key][()] - vbar
            w_tfluc = w_az[t_key][()] - wbar
            b_tfluc = b_az[t_key][()] - bbar

            uu_tfluc_bar += u_tfluc*u_tfluc*dt
            uv_tfluc_bar += u_tfluc*v_tfluc*dt
            uw_tfluc_bar += u_tfluc*w_tfluc*dt
            ub_tfluc_bar += u_tfluc*b_tfluc*dt
            vv_tfluc_bar += v_tfluc*v_tfluc*dt
            vw_tfluc_bar += v_tfluc*w_tfluc*dt
            ww_tfluc_bar += w_tfluc*w_tfluc*dt
            wb_tfluc_bar += w_tfluc*b_tfluc*dt
            bb_tfluc_bar += b_tfluc*b_tfluc*dt

        uu_tfluc_bar /= t_run
        uv_tfluc_bar /= t_run
        uw_tfluc_bar /= t_run
        ub_tfluc_bar /= t_run
        vv_tfluc_bar /= t_run
        vw_tfluc_bar /= t_run
        ww_tfluc_bar /= t_run
        wb_tfluc_bar /= t_run
        bb_tfluc_bar /= t_run

        ufluc2bar = uu_sfluc_bar + uu_tfluc_bar
        uflucvflucbar = uv_sfluc_bar + uv_tfluc_bar
        uflucwflucbar = uw_sfluc_bar + uw_tfluc_bar
        uflucbflucbar = ub_sfluc_bar + ub_tfluc_bar
        vfluc2bar = vv_sfluc_bar + vv_tfluc_bar
        vflucwflucbar = vw_sfluc_bar + vw_tfluc_bar
        wfluc2bar = ww_sfluc_bar + ww_tfluc_bar
        wflucbflucbar = wb_sfluc_bar+wb_tfluc_bar
        bfluc2bar = bb_sfluc_bar+bb_tfluc_bar

        data_dict['ufluc2'] = ufluc2bar
        data_dict['uflucvfluc'] = uflucvflucbar
        data_dict['uflucwfluc'] = uflucwflucbar
        data_dict['uflucbfluc'] = uflucbflucbar
        data_dict['vfluc2'] = vfluc2bar
        data_dict['vflucwfluc'] = vflucwflucbar
        data_dict['wfluc2'] = wfluc2bar
        data_dict['wflucbfluc'] = wflucbflucbar
        data_dict['bfluc2'] = bfluc2bar

    return data_dict

def g2gf_1d(md, var):
    dx = md['LX']/md['Nx']

    var_rft = np.fft.rfft(var, axis=-1)
    xfreq = np.fft.rfftfreq(md['Nx']) * 2.j * np.pi * md['Nx'] / md['LX']

    kernel = np.exp(xfreq * dx/2)
    var_irft = np.fft.irfft(kernel * var_rft, axis=-1)

    return var_irft

def compute_F0(f, md, tstart_ind = 0, verbose=True, tracer=False):
    dr = md['LX']/md['Nx']
    nbins = int(md['Nx']/2)
    r_bins = np.array([r*dr for r in range(0, nbins+1)])
    r_points = np.array([0.5*(r_bins[i]+r_bins[i+1]) for i in range(nbins)])

    data = get_az_data(join(f,'az_stats.h5'), md, tstart_ind = tstart_ind, verbose=verbose)
    wbar = data['w']
    bbar = data['b']
    tbar = data['th']

    if tracer:
        F0_int = 2*integrate.trapezoid(r_points * tbar * wbar, x=r_points, axis=-1)
    else:
        F0_int = 2*integrate.trapezoid(r_points * bbar * wbar, x=r_points, axis=-1)

    _, _, gzf, _ = get_grid(join(f,'grid.h5'), md)

    z_bottom = get_index(md['Lyc'] + md['Lyp'], gzf)
    z_top = get_index(0.9*md['H'], gzf)

    return np.mean(F0_int[z_bottom:z_top])
