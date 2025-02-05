import numpy as np
import warnings

def waveshoal_h_nonmono(T, H0, theta0, gamma, h):
    """
    Code modified from M.Moulton on April 2, 2023
    """
    
    # Suppress RuntimeWarning
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    # Constants
    g = 9.81  # m/s^2

    # Calculate wavelengths in deep water at depths h

    # Deep water wavelength:
    Ldeep = g * T**2 / (2 * np.pi)

    # Wavelength, Ole Madsen approx:
    L = Ldeep * (1 - np.exp(-(2 * np.pi * h / Ldeep)**1.25))**0.4
    

    # Calculate group and phase speeds at depths h
    c = L / T  # Phase speed
    k = 2 * np.pi / L  # Wavenumber
    cg = (L / (2 * T)) * (1 + 2 * (k) * h / np.sinh(2 * (k) * h))  # Group velocity

    # Calculate group and phase speeds at depth h0
    c0 = c[0]  # Phase speed at depth h0
    cg0 = cg[0]  # Phase speed at depth h0

    # Compute wave height and angle at depths h
    theta = np.arcsin(c / c0 * np.sin(np.radians(theta0)))
    H = H0 * np.sqrt(cg0 / cg) * np.sqrt(np.cos(np.abs(np.radians(theta0))) / np.cos(np.abs(np.radians(theta))))

    # Calculate breaking variables
    breaking_index = np.where(H / h > gamma)[0]
    breaking_index = breaking_index[0]
    breaking_depth = h[breaking_index]
    breaking_height = H[breaking_index]
    breaking_angle = np.degrees(theta[breaking_index])

    # NaN values above shoreline
    theta[h < 0] = np.nan
    H[h < 0] = np.nan
    L[h < 0] = np.nan
    c[h < 0] = np.nan
    cg[h < 0] = np.nan

    # Store variables
    wave = {'h': h,
            'L': L,
            'Ldeep': Ldeep,
            'H': H,
            'c': c,
            'cg': cg,
            'theta': np.degrees(theta),
            'breaking_depth': breaking_depth,
            'breaking_height': breaking_height,
            'breaking_angle': breaking_angle,
            'breaking_index': breaking_index}

    # Compute profile onshore of breaking
    H[breaking_index] = h[breaking_index] * gamma
    binds = np.arange(breaking_index, len(h) - 1)

    for ii in binds:
        if h[ii + 1] < h[ii]:
            # Depth-limited breaking
            H[ii + 1] = h[ii + 1] * gamma
        else:
            # Re-shoaling)
            h_2 = h[ii:]
            H0_2 = H[ii]
            theta0_2 = np.degrees(theta[ii])

            wave_2 = waveshoal_subf(T, H0_2, theta0_2, gamma, h_2)

            H_2 = wave_2['H']
            theta_2 = wave_2['theta']
            breaking_index2 = wave_2['breaking_index']

            # Fill in values
            H[ii:] = H_2
            theta[ii:] = theta_2

            # Second breaking region
            H[ii + breaking_index2:] = h[ii + breaking_index2:] * gamma

            break

    # NaN values above shoreline
    theta[h < 0] = np.nan
    H[h < 0] = np.nan

    wave['H'] = H
    wave['theta'] = np.degrees(theta)
    
    # AReset the warnings
    warnings.resetwarnings()

    return wave

def waveshoal_subf(T, H0, theta0, gamma, h):
    # Constants
    g = 9.81  # m/s^2

    # Calculate wavelengths in deep water at depths h

    # Deep water wavelength:
    Ldeep = g * T**2 / (2 * np.pi)

    # Wavelength, Ole Madsen approx:
    L = Ldeep * (1 - np.exp(-(2 * np.pi * h / Ldeep)**1.25))**0.4

    # Calculate group and phase speeds at depths h
    c = L / T  # Phase speed
    k = 2 * np.pi / L  # Wavenumber
    cg = (L / (2 * T)) * (1 + 2 * (k) * h / np.sinh(2 * (k) * h))  # Group velocity

    # Calculate group and phase speeds at depth h0
    c0 = c[0]  # Phase speed at depth h0
    cg0 = cg[0]  # Phase speed at depth h0

    # Compute wave height and angle at depths h
    theta = np.arcsin(c / c0 * np.sin(np.radians(theta0)))
    H = H0 * np.sqrt(cg0 / cg) * np.sqrt(np.cos(np.abs(np.radians(theta0))) / np.cos(np.abs(np.radians(theta))))

    # NaN values above shoreline
    theta[h < 0] = np.nan
    H[h < 0] = np.nan
    L[h < 0] = np.nan
    c[h < 0] = np.nan
    cg[h < 0] = np.nan

    # Calculate breaking variables
    breaking_index = np.where(H / h > gamma)[0]
    breaking_index = breaking_index[0]
    breaking_depth = h[breaking_index]
    breaking_height = H[breaking_index]
    breaking_angle = np.degrees(theta[breaking_index])

    # Store variables
    wave_sub = {'h': h,
                'L': L,
                'Ldeep': Ldeep,
                'H': H,
                'c': c,
                'cg': cg,
                'theta': np.degrees(theta),
                'breaking_depth': breaking_depth,
                'breaking_height': breaking_height,
                'breaking_angle': breaking_angle,
                'breaking_index': breaking_index}

    return wave_sub

def dispersi(const):
    """
    Solve the linear dispersion relationship for surface gravity waves.

    Input:
        const = omega^2 * h / g, where:
            omega = radian frequency,
            h = water depth,
            g = acceleration due to gravity.
    Output:
        kh = wavenumber times water depth.
    Notes: Solves omega^2 * h / g = kh * tanh(kh) using Newton-Raphson iteration.
    
    Code to compute omega from frequency or period:
          g = 9.81
          omega = 2 * np.pi * f = 2 * np.pi / T
          const = (omega ** 2) * h / g
    Code to compute wavelength from kh:
          k = kh / h
          L = 2 * np.pi / k

    Code modified by C.M.Baker on Oct 2024
    """

    # Convert input to array, ensuring it's at least 1D
    const = np.atleast_1d(const)

    # Set negative values to NaN to avoid sqrt warning, assuming they are invalid
    const = np.where(const < 0, np.nan, const)

    # Initialize kh with zeros where const is zero, else with sqrt(const) as initial guess
    kh = np.where(const == 0, 0, np.sqrt(const))

    # Newton-Raphson iteration
    tolerance = 1e-6
    max_iterations = 100
    for _ in range(max_iterations):
        delta = (kh * np.tanh(kh) - const) / (np.tanh(kh) + kh / np.cosh(kh)**2)
        kh -= delta
        if np.all(np.abs(delta) < tolerance):
            break

    return kh

def calc_omega_k(f, h):
    """ 
    f = frequency
    h = water depth 
    
    Example usage:
    [omega, k] = calc_omega_k(frequency_value, water_depth_value)
    """
    
    # Constants
    g = 9.81
    
    # Angular frequency calculation
    omega = 2 * np.pi / (f ** (-1))
    
    # Constants calculation
    cons = omega ** 2 * h / g
    
    # Call to dispersi function (assuming it's defined elsewhere)
    kh = dispersi(cons)
    
    # Calculate k
    k = kh / h
    
    return omega, float(k)

# Compute horizontal and vertical velocity components
def orbital_velocity(a, omega, k, z, h, t, x, phi):
    """ Computes horizontal and vertical velocities for a given wave component. """
    A = a * np.cosh(k * (z + h)) / np.sinh(k * h)
    B = a * np.sinh(k * (z + h)) / np.sinh(k * h)
    u = omega * A * np.cos(k * x - omega * t + phi)
    w = omega * B * np.sin(k * x - omega * t + phi)
    return u, w, A, B

def define_profile(lab_slope, lab_depth, lab_beach, lab_length, swl):
    # Setup bathymetry
    dx = 0.1
    x = np.arange(0, lab_length + dx, dx)
    d = np.zeros_like(x)

    # Offshore slope
    slope_off = 1 / 43.5
    off_depth = ft2m(66.7) * slope_off
    ioffbeach = indices_between(x, ft2m(11.11 + 0.31), ft2m(11.11 + 0.31 + 66.7))
    d[ioffbeach] = np.linspace(0, off_depth, len(ioffbeach))

    # Transition to the lab slope
    ibeach_start = ioffbeach[-1] + 1  # Start right after offshore slope
    ibeach = np.arange(ibeach_start, len(x), 1)

    # Generate the lab slope ensuring matching array sizes
    slope_profile = np.linspace(d[ibeach_start - 1] + dx * lab_slope, 
                                d[ibeach_start - 1] + (len(ibeach) - 1) * dx * lab_slope, 
                                len(ibeach))

    d[ibeach] = slope_profile  # Assign smoothly varying depths

    # Ensure continuity at the transition
    d[:ioffbeach[0]] = 0  # Flat region before offshore slope
    d[ibeach_start] = d[ibeach_start - 1]  # Smooth transition

    # Define water depth
    h = -(d - swl)

    # Window location
    winloc = ft2m(115.41 + 0.31 + np.array([0, 48]))
     
    return d, h, winloc

def calc_LH_boundwave(a1, a2, f1, f2, h, t):
    rho = 997
    g = 9.81
    
    # Average wave statistics
    T = 2 / (f1 + f2)  # Peak wave period
    omega0, k0 = calc_omega_k((f1 + f2)/2, h)  # Angular frequency and wavenumber of bichromatic waves
    L = 2 * np.pi / k0  # Wavelength
    c = L / T
    
    # statistics of individual waves
    [omega1, k1] = calc_omega_k(f1, h)
    [omega2, k2] = calc_omega_k(f2, h)
    
    # group stats
    omegag = abs(omega1 - omega2)
    kg = abs(k1 - k2)
    fg = abs(f1 - f2)  # Group frequency
    cg = omegag / kg
    Eg_t = 0.5 * rho * g * ((a1 + a2) * np.cos(0.5 * omegag * t)) ** 2 # energy of group as a fn of time
    Sxx_t = Eg_t * (( 2 * cg / c ) - 0.5) # radiation stress of group as a fn of time
    ag_t = - (1/rho) * Sxx_t / (g * h - cg**2) # amplitude of bound wave as a fn of time
    
    return ag_t

def find_closest_index(lst, target):
    closest_index = min(range(len(lst)), key=lambda i: abs(lst[i] - target))
    return closest_index

def indices_between(lst, lower_bound, upper_bound):
    between_indices = [i for i, x in enumerate(lst) if lower_bound < x < upper_bound]
    return between_indices

def ft2m(feet):
    meters = feet * 0.3048  # 1 foot is approximately equal to 0.3048 meters
    return meters