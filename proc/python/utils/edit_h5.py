import h5py

editdir = '/home/cwp29/diablo3/strat/high_res2/'
editfile = 'az_stats.h5'

"""
with h5py.File(editdir+editfile, 'a') as f:
    for key in f.keys():
        times = []
        for tkey in f[key].keys():
            times.append(f[key][tkey].attrs['Time'])

        for i in range(1,len(times)):
            if times[i] < times[i-1]:
                trunc_idx = i

        t_delete = list(f[key].keys())
        t_delete = t_delete[:trunc_idx]

        for tkey in t_delete:
            del f[key][tkey]
            """

with h5py.File(editdir+editfile, 'a') as f:
    for key in f.keys():
        times = []
        tkeys = list(f[key].keys())
        for tkey in tkeys:
            times.append(f[key][tkey].attrs['Time'])

        t_delete = []
        for i in range(1,len(times)):
            if times[i] < 2:
                t_delete.append(tkeys[i])


        for tkey in t_delete:
            del f[key][tkey]
