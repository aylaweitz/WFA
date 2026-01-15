import numpy as np
from tqdm import tqdm



def find_lambda_0(data, wavelengths):
    """
    Find the minimum of the line core separately for each spatial position.

    data can be:
      - 2D (wavelength, x)
      - 3D (wavelength, x, y)
      - 4D (wavelength, x, y, …)

    Returns array λ₀ matching the spatial dimensions.
    """
    data = np.asarray(data)
    # i0, i1 = int(wave_range[0]), int(wave_range[1])

    # index of min along wavelength axis but only within slice
    slice_min_index = np.argmin(data, axis=0)

    return wavelengths[slice_min_index]


###### WFA -- for just Ca II right now
g = 1.1
l0 = 8452
C = 4.66*10**(-13) * g * l0**2

def compute_B_parallel(wavelengths, I, V, C, lambda_range):
    """
    Compute B_parallel from arrays wave (wavelength), I(lambda), V(lambda) and constant C:
        B = - sum(dI/dλ * V) / ( C * sum( (dI/dλ)**2 ) )
    """

    B_par = np.empty(I[0].shape)

    center_positions = find_lambda_0(I, wavelengths)
    
    for x in tqdm(range(I[0].shape[0])): # deal with each spatial position seperately -- unique center position
        for y in range(I[0].shape[1]):

            lambda_0 = center_positions[x, y]

            offset = wavelengths - lambda_0
    
            lambda_min = np.min(lambda_range)
            lambda_max = np.max(lambda_range)
    
            mask = (offset >= lambda_min) & (offset <= lambda_max)

            
            wave = np.asarray(wavelengths[mask], dtype=float)
            I_sel = np.asarray(I[mask, x, y], dtype=float)
            V_sel = np.asarray(V[mask, x, y], dtype=float)
        
            # numerical derivative
            dIdl = np.gradient(I_sel, wavelengths, axis=0)
            
            numerator = np.sum(dIdl * V_sel, axis=0)
            denominator = np.sum((dIdl**2), axis=0)
        
            B_par_val = - numerator / (C * denominator)

            B_par[x,y] = B_par_val

    return B_par






# for transverse field
l0 = 8542.1
G = 1.18
# CT = (4.6686 * 10**(-10) * l0**2)**2 * G
CT = (4.67e-13*l0**2)**2*G

def compute_B_perp(wavelengths,
                   I,
                   Q,
                   U,
                   V,
                   C_perp = CT,
                   lambda_0=l0,
                   lambda_range=(-0.4, -0.1)
                  ):
    """
    Compute B_perp from discrete wavelength & intensity arrays according to:
        B_perp = sqrt( ((4/3) * (1/C_perp) * sum(L |1/(λ-λ_w)| * |dI/dλ|))
                        / sum(|1/(λ-λ_w)|^2 * |dI/dλ|^2) )
                        
    *** use l - l0 = [-0.4, -0.1] (centeno 2018)
    """

    B_perp = np.empty(I[0].shape)

    center_positions = find_lambda_0(I, wavelengths)
    
    for x in tqdm(range(I[0].shape[0])): # deal with each spatial position seperately -- unique center position
        for y in range(I[0].shape[1]):

            lambda_0 = center_positions[x, y]
    
            wavelengths = np.asarray(wavelengths, dtype=float)
            I_pos = np.asarray(I[:, x, y], dtype=float)
            Q_pos = np.asarray(Q[:, x, y], dtype=float)
            U_pos = np.asarray(U[:, x, y], dtype=float)
    
            L = np.sqrt(Q_pos**2 + U_pos**2)
    
            # Numerical derivative dI/dλ
            dI = np.gradient(I_pos, wavelengths, axis=0)
    
            # Select points within desired offset range (e.g., -0.4 ≤ λ−λ₀ ≤ -0.1)
            offset = wavelengths - lambda_0
    
            lambda_min = np.min(lambda_range)
            lambda_max = np.max(lambda_range)
    
            mask = (offset >= lambda_min) & (offset <= lambda_max)
    
            lam_sel = wavelengths[mask]
            dI_sel = dI[mask]   # keeps shape (n_selected, n_profiles)
            L_sel = L[mask]
        
            abs_inv = np.abs(1 / (lam_sel - lambda_0))#[:, None, None]
            abs_dI = np.abs(dI_sel)
        
            numerator = (4/3) * (1/ C_perp) * np.sum(L_sel * abs_inv * abs_dI)#, axis=0)
            denominator = np.sum(abs_inv**2 * abs_dI**2)#, axis=0)
        
            B_perp_val = np.sqrt(numerator / denominator)
    
            # print(B_perp_val)
    
            B_perp[x,y] = B_perp_val

    
    return B_perp



def compute_azimuth(wavelengths,
                    I,
                    Q,
                    U,
                    lambda_range=(-0.4, -0.1)
                   ):


    azimuth = np.empty(I[0].shape)

    center_positions = find_lambda_0(I, wavelengths)
    
    for x in tqdm(range(I[0].shape[0])): # deal with each spatial position seperately -- unique center position
        for y in range(I[0].shape[1]):

            lambda_0 = center_positions[x, y]
    
            # wavelengths = np.asarray(wavelengths, dtype=float)
            # # I_pos = np.asarray(I[:, x, y], dtype=float)
            # Q_pos = np.asarray(Q[:, x, y], dtype=float)
            # U_pos = np.asarray(U[:, x, y], dtype=float)
    
            # Select points within desired offset range (e.g., -0.4 ≤ λ−λ₀ ≤ -0.1)
            offset = wavelengths - lambda_0
    
            lambda_min = np.min(lambda_range)
            lambda_max = np.max(lambda_range)
    
            mask = (offset >= lambda_min) & (offset <= lambda_max)

            lam_sel = wavelengths[mask]
            Q_sel = Q[mask, x, y]
            U_sel = U[mask, x, y]

            azimuth_val = 1/2 * np.atan2(np.sum(U_sel), np.sum(Q_sel)) # radians
            azimuth_deg = np.rad2deg(azimuth_val)

            azimuth[x,y] = azimuth_deg

    return azimuth

    