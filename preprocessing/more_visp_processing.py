import numpy as np
from tqdm import tqdm
from scipy.signal import medfilt

def across_slit_correction(caii_data, KERNEL=55): # smoothing by 1 angstrom

    caii_new = np.empty(caii_data.shape)
    means = np.empty((4, caii_data.shape[1], caii_data.shape[3]))
    smoothed = np.empty((4, caii_data.shape[1], caii_data.shape[3]))
    
    for wave in tqdm(range(caii_data.shape[1])): # for each wavelength

        # caii_data is organized = [stokes, wave, y, x]
        
        i = caii_data[0, wave, :, :] / np.median(caii_data[0, wave]) # normalize by median intensity at that wavelength over all pos
        q = caii_data[1, wave, :, :]
        u = caii_data[2, wave, :, :]
        v = caii_data[3, wave, :, :]

        # mean along slit for each wavelength
        mean_i_across = np.median(i, axis=0) #/ np.median(i, axis=0) 
        mean_q_across = np.median(q, axis=0)
        mean_u_across = np.median(u, axis=0)
        mean_v_across = np.median(v, axis=0)

        # add to means array
        means[0, wave] = mean_i_across
        means[1, wave] = mean_q_across
        means[2, wave] = mean_u_across
        means[3, wave] = mean_v_across

    
    smoothed_i = np.apply_along_axis(medfilt, 0, means[0], kernel_size=KERNEL) # default smoothing by ~ 1 angstrom 
    smoothed_q = np.apply_along_axis(medfilt, 0, means[1], kernel_size=KERNEL)
    smoothed_u = np.apply_along_axis(medfilt, 0, means[2], kernel_size=KERNEL)
    smoothed_v = np.apply_along_axis(medfilt, 0, means[3], kernel_size=KERNEL) # has shape [wavelength, along slit] 

    smoothed[0] = smoothed_i
    smoothed[1] = smoothed_q
    smoothed[2] = smoothed_u
    smoothed[3] = smoothed_v

    for wave in tqdm(range(caii_data.shape[1])): # for each wavelength

        # get the stokes param at each wavelength
        i = caii_data[0, wave, :, :] #/ np.median(caii_data[0, wave]) # normalize by median intensity at that wavelength over all pos
        q = caii_data[1, wave, :, :]
        u = caii_data[2, wave, :, :]
        v = caii_data[3, wave, :, :]

        # for each corresponding wavelength, subtract off the smoothed correction
        subtracted_i = i / smoothed_i[wave, :]
        subtracted_q = q - smoothed_q[wave, :]
        subtracted_u = u - smoothed_u[wave, :]
        subtracted_v = v - smoothed_v[wave, :]
    
        # add to new corrected data array
        caii_new[0, wave] = subtracted_i 
        caii_new[1, wave] = subtracted_q
        caii_new[2, wave] = subtracted_u
        caii_new[3, wave] = subtracted_v
    
    return caii_new, means, smoothed


##########################################################


def along_slit_correction(caii_data, KERNEL=55): # smoothing by ??

    caii_new = np.empty(caii_data.shape) # stokes, wave, scan direction, slit direction
    means = np.empty((4, caii_data.shape[1], caii_data.shape[2]))
    smoothed = np.empty((4, caii_data.shape[1], caii_data.shape[2]))

    for wave in tqdm(range(caii_data.shape[1])): # for each wavelength

        # caii_data is organized = [stokes, wave, y, x]
        
        i = caii_data[0, wave, :, :] / np.median(caii_data[0, wave]) # normalize by median intensity at that wavelength over all pos
        q = caii_data[1, wave, :, :]
        u = caii_data[2, wave, :, :]
        v = caii_data[3, wave, :, :]

        # mean along slit for each wavelength
        mean_i_along = np.median(i, axis=1) #/ np.median(i, axis=0) 
        mean_q_along = np.median(q, axis=1)
        mean_u_along = np.median(u, axis=1)
        mean_v_along = np.median(v, axis=1)

        # add to means array
        means[0, wave] = mean_i_along
        means[1, wave] = mean_q_along
        means[2, wave] = mean_u_along
        means[3, wave] = mean_v_along

    # fit polynomial instead of just taking mean? the fitting with kind of smooth?
    smoothed_i = np.apply_along_axis(medfilt, 0, means[0], kernel_size=KERNEL) # default smoothing by ~ 1 angstrom 
    smoothed_q = np.apply_along_axis(medfilt, 0, means[1], kernel_size=KERNEL)
    smoothed_u = np.apply_along_axis(medfilt, 0, means[2], kernel_size=KERNEL)
    smoothed_v = np.apply_along_axis(medfilt, 0, means[3], kernel_size=KERNEL) # has shape [wavelength, along slit] 

    smoothed[0] = smoothed_i
    smoothed[1] = smoothed_q
    smoothed[2] = smoothed_u
    smoothed[3] = smoothed_v

    for wave in tqdm(range(caii_data.shape[1])): # for each wavelength

        # get the stokes param at each wavelength
        i = caii_data[0, wave, :, :] #/ np.median(caii_data[0, wave]) # normalize by median intensity at that wavelength over all pos
        q = caii_data[1, wave, :, :]
        u = caii_data[2, wave, :, :]
        v = caii_data[3, wave, :, :]

        # for each corresponding wavelength, subtract off the smoothed correction
        subtracted_i = i / smoothed_i[wave, :, None] # transpose/flip the last 2 axes (spatial)
        subtracted_q = q - smoothed_q[wave, :, None]
        subtracted_u = u - smoothed_u[wave, :, None]
        subtracted_v = v - smoothed_v[wave, :, None]
    
        # add to new corrected data array
        caii_new[0, wave] = subtracted_i 
        caii_new[1, wave] = subtracted_q
        caii_new[2, wave] = subtracted_u
        caii_new[3, wave] = subtracted_v
    
    return caii_new, means, smoothed