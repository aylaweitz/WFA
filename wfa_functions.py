import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field

@dataclass
class SpectralLine:
    name: str
    g_los: float                    # LOS Landé factor
    g_trans: float                  # transverse Landé factor
    lambda0: float                  # central wavelength [angstrom]
    wave_min_search_range: float    # range of wavelengths to search for line core
    lambda_range_los: tuple         # wavelength range to consider for los b field calculation [angstrom from line core]
    lambda_range_perp: tuple        # wavelength range to consider for transverse b field calculation [angstrom from line core]

    C_par: float = field(init=False)
    C_trans: float = field(init=False)

    def __post_init__(self):
        self.lambdaB = 4.6686*10**(-13) # [Å] zeeman splitting coeff -- polarization of spectral lines (3.14)
        self.C_par = self.lambdaB * self.lambda0**2 * self.g_los # should this change with where the line core is at each position?
        self.C_trans = (self.lambdaB * self.lambda0**2)**2 * self.g_trans
        


CaII_8542 = SpectralLine(
    name = "Ca II 8542", ## all parameters from Centeno 2018 https://ui.adsabs.harvard.edu/abs/2018ApJ...866...89C/abstract 
    g_los = 1.10,
    g_trans = 1.18,
    lambda0 = 8542,
    wave_min_search_range = (8540.0, 8545.0),
    lambda_range_los = (-0.25, 0.25),
    lambda_range_perp = (-0.4, -0.1)
)

NaI_D1_5896 = SpectralLine(
    name = "Na I D1 5896",
    g_los = 1.1, # PLACEHOLDER 1 https://steck.us/alkalidata/sodiumnumbers.1.6.pdf 
    g_trans = 1.1, # PLACEHOLDER 1 metcalf 1995
    lambda0 = 5896,
    wave_min_search_range = (5895.0, 5897.0),
    lambda_range_los = (-0.25, 0.25),
    lambda_range_perp = (-0.25, -0.1)
)

FeI_6302 = SpectralLine( # lets use 6302 line
    name = "Fe I 6302",
    g_los = 2.5, # (g = 1.667 for Fe i 6301.5 Å, g = 2.5 for Fe i 6302.5 Å) -- https://www.aanda.org/articles/aa/pdf/2010/09/aa13972-09.pdf 
    g_trans = 1, # PLACEHOLDER 1 check out https://pubs.aip.org/aip/jpr/article/4/2/353/242018/Energy-levels-of-iron-Fe-I-through-Fe-XXVI 
    lambda0 = 6302,
    wave_min_search_range = (6302.25, 6302.75),
    lambda_range_los = (-0.25, 0.25),
    lambda_range_perp = (-0.25, -0.1)
)


def find_lambda_0(data, wavelengths, line):
    """
    Find the minimum of the line core separately for each spatial position,
    searching only within a given wavelength range.

    Parameters
    ----------
    data : ndarray
        Shape (wavelength, x[, y, ...])
    wavelengths : ndarray
        1D array of wavelengths, length = wavelength axis of data
    wave_range : tuple
        (lambda_min, lambda_max)

    Returns
    -------
    lambda_0 : ndarray
        Array of λ₀ matching the spatial dimensions of data.
    """
    data = np.asarray(data)
    wavelengths = np.asarray(wavelengths)

    # Create mask for wavelength range
    mask = (wavelengths >= line.wave_min_search_range[0]) & (wavelengths <= line.wave_min_search_range[1])

    if not np.any(mask):
        raise ValueError("No wavelengths found in the specified range.")

    # Slice data and wavelengths
    data_slice = data[mask, ...]
    wavelengths_slice = wavelengths[mask]

    # Index of minimum along wavelength axis (within range)
    slice_min_index = np.argmin(data_slice, axis=0)

    # Convert indices to actual wavelengths
    return wavelengths_slice[slice_min_index]


def compute_B_parallel(wavelengths, I, V, line):
    """
    Compute B_parallel from arrays wave (wavelength), I(lambda), V(lambda) and constant C:
        B = - sum(dI/dλ * V) / ( C * sum( (dI/dλ)**2 ) )
    """

    B_par = np.empty(I[0].shape)

    center_positions = find_lambda_0(I, wavelengths, line)
    
    for x in tqdm(range(I[0].shape[0])): # deal with each spatial position seperately -- unique center position
        for y in range(I[0].shape[1]):

            lambda_0 = center_positions[x, y]

            offset = wavelengths - lambda_0
    
            lambda_min = np.min(line.lambda_range_los)
            lambda_max = np.max(line.lambda_range_los)

            # print(lambda_min, lambda_max)
    
            mask = (offset >= lambda_min) & (offset <= lambda_max)

            
            wave = np.asarray(wavelengths[mask], dtype=float)
            I_sel = np.asarray(I[mask, x, y], dtype=float)
            V_sel = np.asarray(V[mask, x, y], dtype=float)
        
            # numerical derivative
            dIdl = np.gradient(I_sel, wave, axis=0) # changed from 'wavelengths' to 'wave'
            
            numerator = np.sum(dIdl * V_sel, axis=0)
            denominator = np.sum((dIdl**2), axis=0)
        
            B_par_val = - numerator / (line.C_par * denominator)

            B_par[x,y] = B_par_val

    return B_par


def compute_B_perp(wavelengths,
                   I,
                   Q,
                   U,
                   V,
                   line
                  ):
    """
    Compute B_perp from discrete wavelength & intensity arrays according to:
        B_perp = sqrt( ((4/3) * (1/C_perp) * sum(L |1/(λ-λ_w)| * |dI/dλ|))
                        / sum(|1/(λ-λ_w)|^2 * |dI/dλ|^2) )
                        
    *** use l - l0 = [-0.4, -0.1] (centeno 2018)
    """

    B_perp = np.empty(I[0].shape)

    center_positions = find_lambda_0(I, wavelengths, line)
    
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
    
            lambda_min = np.min(line.lambda_range_perp)
            lambda_max = np.max(line.lambda_range_perp)

            # print(lambda_min, lambda_max)
    
            mask = (offset >= lambda_min) & (offset <= lambda_max)
    
            lam_sel = wavelengths[mask]
            dI_sel = dI[mask]   # keeps shape (n_selected, n_profiles)
            L_sel = L[mask]
        
            abs_inv = np.abs(1 / (lam_sel - lambda_0))#[:, None, None]
            abs_dI = np.abs(dI_sel)
        
            numerator = (4/3) * (1/ line.C_trans) * np.sum(L_sel * abs_inv * abs_dI)#, axis=0)
            denominator = np.sum(abs_inv**2 * abs_dI**2)#, axis=0)
        
            B_perp_val = np.sqrt(numerator / denominator)
    
            # print(B_perp_val)
    
            B_perp[x,y] = B_perp_val

    
    return B_perp


# pretty sure involves no info about the spectral line itself -- only Q, U ratio
def compute_azimuth(wavelengths,
                    I,
                    Q,
                    U,
                    line
                   ):


    azimuth = np.empty(I[0].shape)

    center_positions = find_lambda_0(I, wavelengths, line)
    
    for x in tqdm(range(I[0].shape[0])): # deal with each spatial position seperately -- unique center position
        for y in range(I[0].shape[1]):

            lambda_0 = center_positions[x, y]
    
            # wavelengths = np.asarray(wavelengths, dtype=float)
            # # I_pos = np.asarray(I[:, x, y], dtype=float)
            # Q_pos = np.asarray(Q[:, x, y], dtype=float)
            # U_pos = np.asarray(U[:, x, y], dtype=float)
    
            # Select points within desired offset range (e.g., -0.4 ≤ λ−λ₀ ≤ -0.1)
            offset = wavelengths - lambda_0
    
            lambda_min = np.min(line.lambda_range_perp)
            lambda_max = np.max(line.lambda_range_perp)

            # print(lambda_min, lambda_max)
    
            mask = (offset >= lambda_min) & (offset <= lambda_max)

            # lam_sel = wavelengths[mask]
            Q_sel = Q[mask, x, y]
            U_sel = U[mask, x, y]

            azimuth_val = 1/2 * np.atan2(np.sum(U_sel), np.sum(Q_sel)) # radians --- DO THE ABSOLUTE VALUE
            azimuth_deg = np.rad2deg(azimuth_val)

            azimuth[x,y] = azimuth_deg

    return azimuth