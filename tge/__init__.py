import numpy as np
from scipy.special import jv
from numba import njit, prange

k_B = 1.380649e3 # Jy . m^2 / K
c = 299792458.0    # m / s

class TGE():

    def __init__(self, dish_d: float, ref_freq: float, f: float, N_grid_x: int):
        """Tapered Gridded Estimator of the Angular Power Spectrum.

        Parameters:
        -----------
        dish_d: float
            Dish diameter in metres. Used to calculate the solid angle of the beam.
        ref_freq: float
            Reference frequency of the observed visibilities in Hz.
        f: float
            The fraction in relation to
        """
        self.dish_d = dish_d
        self.ref_freq = ref_freq
        self.lamda = c / ref_freq
        self.dBdT = k_B / self.lamda**2
        self.__dict__.update(beam_constants(dish_d, ref_freq, self.dBdT, f))
        self.delta_U = np.sqrt(np.log(2)) / ( 2 * np.pi * self.thetaW )
        self.fov = 1. / self.delta_U
        self.N_side = N_grid_x
        self.lm_g, theta_rr = sky_coords(N_grid_x, self.fov)
        self.uv_g, self.U_g = fourier_coords(N_grid_x, self.fov)

        self.K1g, self.K2gg = None, None
        self.V_cg, self.B_cg, self.E_g = None, None, None

    def estimate_Cl_g(self, uv: np.ndarray, V: np.ndarray):
        """Grid the visibilities and use the TGE to estimate the Angular Power 
        Spectrum on the uv grid.

        Parameters:
        -----------
        uv: np.ndarray (n_vis, 2)
            The uv points of the measured visibilities in units of lambda.
        V: np.ndarray - complex (n_vis,)
            The visibilites.

        Returns:
        --------
        E_g: np.ndarray (n_u*n_v, 2)
            The angular power spectrum estimate at the uv grid points.
        """
        
        self.V_cg, self.K1g = grid(uv, V, self.uv_g, self.U0)
        self.B_cg, self.K2gg = grid_Pk(uv, np.abs(V)**2, self.uv_g, self.U0)
        self.E_g = (np.abs(self.V_cg)**2 - self.B_cg) / (self.K1g**2 * self.V1 - self.K2gg * self.V0)

        return self.E_g

    def estimate_Cl(self, uv: np.ndarray, V: np.ndarray, sigma_n: float, 
                    n_bins: int=10, log_bins: bool=True, regrid: bool=False):
        """Calculate the Angular Power Spectrum binned along l.

        Parameters:
        -----------
        uv: np.ndarray (n_vis, 2)
            The uv-coordinates of each visibility point in units of $\\lambda$.
        V: complex np.ndarray (n_vis,)
            Visibilities in Jy.

        Returns:
        --------
        l: np.ndarray (n_bins,)
            The l-coordinate for each bin in l-space.
        Cl_normed: np.ndarray (n_bins,)
            The expectation of the normalised angular power spectrum (l(l+1)$C_l$/2$\\pi$) in each bin.
        delta_Cl_normed: np.ndarray (n_bins,)
            The standard error of the normalised angular power spectrum in each bin.
        """

        self.sigma_n = sigma_n

        if regrid==True or self.V_cg is None:
            self.estimate_Cl_g(uv, V)
    
        U_bin_edges = np.logspace(np.log10(self.U_g[self.U_g>0].min()), np.log10(self.U_g.max()), n_bins+1)

        weights = self.K1g**2
        weights = np.ones_like(self.K1g)
        
        self.U_b, self.Cl_b = bin_Pk(U_bin_edges, self.U_g.flatten(), self.E_g, weights)
        self.delta_Cl_b = bin_Pk_sig(U_bin_edges, self.uv_g, self.E_g, self.sigma1, 
                                self.V1, self.K1g, self.K2gg, self.sigma_n, 
                                weights)
        
        self.l_b = 2 * np.pi * self.U_b
        self.Cl_norm = self.l_b * (self.l_b + 1) / (2 * np.pi)
    
        return self.l_b, self.Cl_b*self.Cl_norm, self.delta_Cl_b*self.Cl_norm

# def Pk(k, P0=1e10, k0=1e1, gamma=0):
def Pk(k, P0=1e12, k0=1e1, gamma=2.34):
    """Power spectrum model which is flat for small k and power law for 
    large k with the change point defined by 'k0'.
    
    Parameters:
    -----------
    k: np.ndarray
        The k values to evaluate the power spectrum at.
    P0: float
        Power at the smallest k values except k=0.
    k0: float
        k-value at which the power spectrum changes from flat to power law.
    gamma: float
        Power law exponent. 

    Returns:
    --------
    P_k: np.ndarray
        The power spectrum values for each k.
    """
    return np.where(k == 0, 0, P0 / (1 + np.abs(k) / k0) ** gamma)


def Cl(l, A=513e-6, beta=2.34):
    """Angular power spetrum model as a pure power law.

    Parameters:
    -----------
    l: np.ndarray
        The l-values to evaluate the power spectrum at.
    A: float
        The reference power spectrum value at l=1000.
    beta: float
        The power law exponent.

    Returns:
    --------
    C_l: np.ndarray
        The Angular power spectrum values at each l.
    """
    return np.where(l == 0, 0, A * (1000 / np.abs(l)) ** beta)


def airy(theta, D, freq=None, lamda=None):
    """Airy disk model for a total power primary beam for a given circular dish.

    Parameters:
    -----------
    theta: np.ndarray
        The radial offset from the pointing direction in radians.
    D: float
        The dish diameter in metres.
    freq: float
        Observation frequency in Hz.
    lamda: float
        Observation wavelength in metres.

    Returns:
    --------
    beam: np.ndarray
        The beam values.
    """
    if freq is None and lamda in None:
        print("'lamda' or 'freq' must be given.")
        return
    if freq is not None:
        lamda = c / freq
    x = np.pi * theta * (D / lamda)
    return np.where(theta == 0, 1, (2 * jv(1, x) / x) ** 2)


def airy_fourier(U, D, freq=None, lamda=None):
    """Fourier transfrom of the Airy disk primary beam.

    Parameters:
    -----------
    U: np.ndarray
        The distance in uv space.
    D: float
        The dish diameter in metres.
    freq: float
        Observation frequency in Hz.
    lamda: float
        Observation wavelength in metres.

    Returns:
    --------
    beam_f: np.ndarray
        The beam Fourier values.
    """
    if freq is None and lamda in None:
        print("'lamda' or 'freq' must be given.")
        return
    if freq is not None:
        lamda = c / freq
    D_lam = D / lamda
    return np.where(
        U > D_lam,
        0,
        (8 / np.pi**2 / D_lam**4)
        * (D_lam**2 * np.arcos(U / D_lam) - U * np.sqrt(D_lam**2 - U**2)),
    )


@njit
def gauss(x, A, x0):
    """General Gaussian function.

    Parameters:
    -----------
    x: np.ndarray (n_points,)
        The points at which to evaluate the Gaussian.
    A: float
        Amplitude of the Gaussian.
    x0: float
        The width scale of the Gaussian.

    Returns:
    --------
    y: np.ndarray (n_points,)
        The Gaussian evaluated at the given locations.
    """
    return A * np.exp(-((x / x0) ** 2))


def set_nan(x, frac, seed=123):
    """Set a given fraction of an array to NaN values.

    Parameters:
    -----------
    x: np.ndarray
        The array to set random values to NaNs.
    frac: float
        Fraction of array to change to NaNs.
    seed: int
        Seed for random number generator.

    Returns: 
    --------
    x_: np.ndarray
        Input array with random positions changed to NaN.
    """
    shape = x.shape
    N = x.size
    np.random.seed(seed)
    nan_idx = np.random.permutation(N)[: int(frac * N)]
    x = x.flatten()
    if x.dtype is complex:
        x[nan_idx] = np.nan + 1.0j * np.nan
    else:
        x[nan_idx] = np.nan
    return x.reshape(shape)


def complex_noise(shape: tuple, sigma: float, seed: int=124) -> np.array:
    """Generate samples from CN(0, sigma^2)
    
    Parameters:
    -----------
    shape: tuple
        Shape of the array to generate.
    sigma: float
        The real-valued standard deviation of the complex normal.

    Returns:
    --------
    noise: np.ndarray
        Realisation of complex values.
    """
    np.random.seed(seed)
    noise = (
        sigma / np.sqrt(2) * (np.random.randn(*shape) + 1.0j * np.random.randn(*shape))
    )

    return noise


def beam_constants(D: float, freq: float, dBdT: float=None, f: float=0.8) -> dict:
    """Calculate various constants derived from the antenna dimensions.

    Parameters:
    -----------
    D: float
        Dish diameter in meters.
    freq: float
        Observation frequency in Hz.
    dBdT: float
        Conversion factor from temperature in K to spectral flux density in Jy.
    f: float
        Fraction of the expected FoV to gaussian taper to. A value in [0,1].

    Returns:
    --------
    constants: dict
        Dictionary of various constants derived from the theoretical beam and 
        tapering fraction.
    """
    lamda = c / freq

    if dBdT is None:
        dBdT = 2 * k_B / lamda**2

    thetaFWHM = 1.03 * (lamda / D)  # thetaFWHM = 1.025 * (lamda / D)
    theta0 = 0.6 * thetaFWHM
    U0 = 1.0 / (np.pi * theta0)  # U0 = 0.53 / thetaFWHM
    sigma0 = 0.76 / thetaFWHM

    A_U0 = 1.0 / (np.pi * U0**2)

    thetaW = f * theta0
    UW = 1.0 / (np.pi * thetaW)
    A_UW = 1.0 / (np.pi * UW**2)

    theta0W = (
        f / np.sqrt(1 + f**2) * theta0
    )  # theta_0W = 1. / np.sqrt(1. / theta0**2 + 1. / thetaW**2 )

    sigma1 = np.sqrt(1 + f**2) / f * sigma0

    Omega_AW = np.pi * theta0W**2
    Omega_PSAW = Omega_AW / 2

    Omega_W = np.pi * thetaW**2
    Omega_PSW = Omega_W / 2

    Omega_B = np.pi * theta0**2
    Omega_PS = Omega_B / 2

    V0 = dBdT**2 * Omega_PS
    V1 = dBdT**2 * Omega_PSAW

    constants = {
        "lamda": lamda,
        "dBdT": dBdT,
        "thetaFWHM": thetaFWHM,
        "theta0": theta0,
        "U0": U0,
        "sigma0": sigma0,
        "A_U0": A_U0,
        "thetaW": thetaW,
        "UW": UW,
        "A_UW": A_UW,
        "theta0W": theta0W,
        "sigma1": sigma1,
        "Omega_AW": Omega_AW,
        "Omega_PSAW": Omega_PSAW,
        "Omega_W": Omega_W,
        "Omega_PSW": Omega_PSW,
        "Omega_B": Omega_B,
        "Omega_PS": Omega_PS,
        "V0": V0,
        "V1": V1,
    }

    return constants

def lm_to_radec(lm: np.array, ra0: float, dec0: float) -> np.array:
    """Convert sky coordinates from projected radians to right ascension and declination in degrees. 

    Parameters:
    -----------
    lm: np.array (n_points, 2)
        lm-coordinate positions in projected radians.
    ra0: float
        Right ascension of the phase centre in degrees.
    dec0: float
        Declination of the phase centre in degrees.

    Returns:
    --------
    radec: np.array (n_points, 2)
        Right ascension and declination of the lm-coordinate positions.
    """

    ra0, dec0 = np.deg2rad([ra0, dec0])
    l, m = lm.T
    n = np.sqrt(1 - l**2 - m**2)
    dec = np.arcsin(m * np.cos(dec0) + n * np.sin(dec0))
    ra = ra0 + np.arctan2(l, n * np.cos(dec0) - m * np.sin(dec0))
    
    return np.rad2deg(np.array([ra, dec]).T)

def radec_to_lm(radec, ra0, dec0):
    """Convert sky coordinates from right ascension and declination in degrees to projected radians. 

    Parameters:
    -----------
    radec: np.array (n_points, 2)
        Right ascension and declination of the positions.
    ra0: float
        Right ascension of the phase centre in degrees.
    dec0: float
        Declination of the phase centre in degrees.

    Returns:
    --------
    lm: np.array (n_points, 2)
        lm-coordinate positions in projected radians.
    """

    ra0, dec0 = np.deg2rad([ra0, dec0])
    ra, dec = np.deg2rad(radec).T
    delta_ra = ra - ra0
    l = np.cos(dec) * np.sin(delta_ra)
    m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(delta_ra)

    return np.array([l, m]).T 

def sky_coords(N_side, fov):
    """The sky coordinates for an image.

    Parameters:
    -----------
    N_side: int
        Number of pixels along one side of the image.
    fov: float
        The diameter (side length) of the image in projected radians.

    Returns:
    --------
    theta_xy: np.ndarray (N_side^2, 2)
        lm-coordinates in projected radians for each sky pixel.
    theta_rr: np.ndarray (N_side, N_side)
        Radial distance from the centre of the sky area in projected radians.
    """
    theta_max = fov / 2

    theta_x = np.linspace(-theta_max, theta_max, N_side, endpoint=False)
    theta_y = np.linspace(-theta_max, theta_max, N_side, endpoint=False)
    # dtheta = np.diff(theta_x).min()
    theta_xx, theta_yy = np.meshgrid(theta_x, theta_y)
    theta_rr = np.sqrt(theta_xx**2 + theta_yy**2)

    theta_xy = np.stack([theta_xx.flatten(), theta_yy.flatten()]).T

    return theta_xy, theta_rr


def fourier_coords(N_side, fov):
    """The fourier coordinates for a real-valued image.

    Parameters:
    -----------
    N_side: int
        Number of pixels along one side of the image.
    fov: float
        The diameter (side length) of the image in projected radians.
    
    Returns:
    --------
    kxy: np.ndarray (N_side*(N_side//2+1), 2)
        uv-coordinates in lambda for each Fourier component.
    kkrr: np.ndarray (N_side, N_side)
        Magnitude of the Fourier mode in lambda. 
    """
    dtheta = fov / N_side

    kx = np.fft.rfftfreq(N_side, dtheta)
    ky = np.fft.fftfreq(N_side, dtheta)
    dk = np.diff(kx[kx > 0]).min()
    # dVk = dk**2
    kxx, kyy = np.meshgrid(kx, ky)
    krr = np.sqrt(kxx**2 + kyy**2)

    kxy = np.stack([kxx.flatten(), kyy.flatten()]).T

    return kxy, krr


def simulate_sky(
    N_side, fov, Pk=None, Cl=None, PS_args=None, beam=None, B_args=None, seed=123
):
    """Simulate a sky realisation from a given angular power spectrum.

    Parameters:
    -----------
    N_side: int
        The number of pixels along one side of the image.
    fov: float
        The diameter (side length) of the image in projected radians.
    Pk: function
        Angular power spectrum as a function of 'k' which is called as Pk(k, *PS_args).
    Cl: function
        Angular power spectrum as a function of 'l' which is called as Cl(l, *PS_args).
    PS_args: dict
        The arguments to be passed to the angular power spectrum.
    beam: function
        Primary beam (Power) as a function of projected radial distance from the image centre in radians. Called as beam(r, *B_args)
    B_args: dict
        The arguments to the beam function.
    seed: int
        Random seed for the sky realisation.

    Returns:
    --------
    I: np.ndarray (N_side, N_side)
        Simulated image in units defined by the provided Angular Power Spectrum function.
    theta_xy: np.ndarray (N_side^2, 2)
        lm-coordinates of the sky pixels.
    """
    if Pk is None and Cl is None:
        print("'Pk' or 'Cl' must be given.")
        return

    Omega = fov**2

    theta_xy, theta_rr = sky_coords(N_side, fov)

    kxy, krr = fourier_coords(N_side, fov)

    np.random.seed(seed)

    I_norm = np.random.randn(N_side, N_side)
    V_norm = np.fft.rfft2(I_norm) / N_side  # CN(0,1)

    if Pk is not None:
        if PS_args is None:
            PS = Pk(krr)
        else:
            PS = Pk(krr, **PS_args)
    elif Cl is not None:
        if PS_args is None:
            PS = Cl(2 * np.pi * krr)
        else:
            PS = Cl(2 * np.pi * krr, **PS_args)

    if B_args is None:
        B = beam(theta_rr)
    else:
        B = beam(theta_rr, **B_args)

    norm = "backward"
    V = np.sqrt(Omega * PS) * V_norm
    I = np.fft.irfft2(V, norm=norm)
    I_ = I * B

    return I_, theta_xy


def random_uv(
    N_vis: int,
    spread: float,
    dist: str = "gauss",
    tracks: bool = False,
    seed: int = 999,
):
    """ Generate random positions in the uv-domain based on a distribution.

    Parameters:
    -----------
    N_vis: int
        Number of visibilities to generate positions for.
    spread: float
        Parameter to change the spread fo the distribution of points.
    dist: str
        The type fo distribution to generate radnom samples from.
    tracks: bool
        Whether to create uv-tracks.
    seed: int
        Random number generator seed.
    """

    spread = np.abs(spread)

    if dist == "gauss":
        uv = spread * np.random.randn(N_vis, 2)
    elif dist == "uniform":
        uv = np.random.uniform(-spread, spread, shape=(N_vis, 2))

    return uv


@njit(parallel=True)
def grid(uv, vis, kxy, U0, A=None):
    """Grid the visibilities by convolving them with a window function in Fourier space.
    V_cg term in Eqn (4) from arXiv:1603.02513

    Parameters:
    -----------
    uv: np.ndarray (n_bl*n_t, 2)
        The UV positions of each baseline of the ungridded visibilities.
    vis: np.ndarray (n_bl*n_t)
        The ungridded visibilities.
    kxy: np.ndarray (n_u*n_v, 2)
        The grid locations in UV space onto which the visibilities will be gridded.
    U0: float
        The width of the Gaussian gridding kernel.
    A: float
        The amplitude of the Gaussian gridding kernel. Default is None which leads to a window function with unit amplitude in image space.

    Returns:
    --------
    vis_grid: np.ndarray (n_u*n_v,)
        The gridded visibilities at the grid locations.
    weights: np.ndarray (n_u*n_v,)
        The weights to normalize the gridded visibilities.
    """
    uv = np.where(uv[:, :1] < 0, -uv, uv)
    vis = np.where(uv[:, 0] < 0, np.conjugate(vis), vis)
    if A is None:
        A = 1.0 / (np.pi * U0**2)
    vis_grid = np.zeros(len(kxy), dtype=np.complex128)
    weights = np.zeros(len(kxy))
    for i in prange(len(kxy)):
        dku = np.sqrt(np.sum((uv - kxy[i, :]) ** 2, axis=-1))
        idx = np.where(dku < 3 * U0)[0]
        if len(idx) == 0:
            vis_grid[i] = 0.0
        else:
            dku_ = dku[idx]
            w = gauss(dku_, A, U0)  # \tilde{W}
            weights[i] = np.sum(w)  # K_1g
            vis_grid[i] = np.sum(vis[idx] * w)

    return vis_grid, weights


@njit(parallel=True)
def grid_Pk(uv, P_k, kxy, UW, A=None):
    """Grid the visibilities by convolving them with a window function in Fourier space.
    B_cg term in Eqn (7) from arXiv:1603.02513

    Parameters:
    -----------
    uv: np.ndarray (n_bl*n_t, 2)
        The UV positions of each baseline of the ungridded visibilities.
    P_k: np.ndarray (n_bl*n_t)
        The ungridded visibilities magnitudes squared i.e. |V|^2.
    kxy: np.ndarray (n_u*n_v, 2)
        The grid locations in UV space onto which the visibilities will be gridded.
    U0: float
        The width of the Gaussian gridding kernel.
    A: float
        The amplitude of the Gaussian gridding kernel. Default is None which leads to a window function with unit amplitude in image space.

    Returns:
    --------
    Pk_grid: np.ndarray (n_u*n_v)
        The gridded visibility magnitudes squared at the grid locations.
    weights: np.ndarray (n_u*n_v,)
        The weights to normalize the gridded squared visibility magnitudes.
    """
    uv = np.where(uv[:, :1] < 0, -uv, uv)
    if A is None:
        A = 1.0 / (np.pi * UW**2)
    Pk_grid = np.zeros(len(kxy))
    weights = np.zeros(len(kxy))
    for i in prange(len(kxy)):
        dku = np.sqrt(np.sum((uv - kxy[i, :]) ** 2, axis=-1))
        idx = np.where(dku < 3 * UW)[0]
        if len(idx) == 0:
            Pk_grid[i] = 0.0
        else:
            dku_ = dku[idx]
            w = gauss(dku_, A, UW) ** 2  # \tilde{W}^2
            weights[i] = np.sum(w)  # K_2gg
            Pk_grid[i] = np.sum(P_k[idx] * w)

    return Pk_grid, weights


@njit
def bin_Pk(bin_edges, mag_uv, E_g, weights=None):
    """\\hat{E}_g(a) and \bar{l}_a terms in Eqns (36) and (39) from arXiv:1409.7789
    
    Parameters:
    -----------
    bin_edges: np.ndarray (n_bins+1,)
        The edges of the bins.
    mag_uv: np.ndarray (n_grid,)
        The uv magnitudes at the grid locations.
    E_g: np.ndarray (n_grid,)
        The angular power spectrum estimate at the grid locations.
    weights: np.ndarray (n_grid,)
        Weights to use when averaging estimates within a bin.

    Returns:
    --------
    uv_binned: (n_bins,)
        The average uv magnitude in a bin.
    Pk_binned: (n_bins,)
        The standard error of the angular power spectrum estimator in the l-bins. 
    """
    n_bins = len(bin_edges) - 1
    Pk_binned = np.zeros(n_bins)
    uv_binned = np.zeros(n_bins)
    if weights is None:
        weights = np.ones_like(E_g)
    for i in range(n_bins):
        idx = np.where(
            (mag_uv >= bin_edges[i])
            & (mag_uv < bin_edges[i + 1])
            & (~np.isnan(E_g))
            & (E_g > 0)
        )[0]
        if len(idx) == 0:
            Pk_binned[i] = np.nan
            uv_binned[i] = np.nan
        else:
            Pk_binned[i] = np.sum(E_g[idx] * weights[idx]) / np.sum(weights[idx])
            uv_binned[i] = np.sum(mag_uv[idx] * weights[idx]) / np.sum(weights[idx])

    return uv_binned, Pk_binned


# @njit
# def bin_Pk_sig(bin_edges, nkxy, E_g, sigma1, weights=None):
#     n_bins = len(bin_edges) - 1
#     Pk_sig_binned = np.zeros(n_bins)
#     U = np.sqrt(nkxy[:,0]**2 + nkxy[:,1]**2)
#     if weights is None:
#         weights = np.ones(len(U))
#     for i in range(n_bins):
#         idx = np.where((U>=bin_edges[i]) & (U<bin_edges[i+1]) & (~np.isnan(E_g)) & (E_g>0))[0]
#         if len(idx)==0:
#             Pk_sig_binned[i] = np.nan
#         else:
#             dU = nkxy[idx][:,None,:] - nkxy[idx][None,:,:]
#             dU = np.sqrt(dU[:,:,0]**2 + dU[:,:,1]**2)
#             w_2gg = weights[idx][:,None]*weights[idx][None,:] * np.exp(-2*(dU/sigma1)**2)

#             Pk_sig_binned[i] = np.sqrt(np.sum( w_2gg * E_g[idx]**2 )) / np.sum(weights[idx])

#     return Pk_sig_binned


@njit
def bin_Pk_sig(bin_edges, nkxy, E_g, sigma1, V1, K1g, K2gg, sigma_n, weights=None):
    """\\sigma^2_E_G(a) term in Eqn (42) from arXiv:1409.7789
    
    Parameters:
    -----------
    bin_edges: np.ndarray (n_bins+1,)
        The edges of the bins.
    nkxy: np.ndarray (n_grid,)
        The uv magnitudes at the grid locations.
    E_g: np.ndarray (n_grid,)
        The angular power spectrum estimate at the grid locations.
    V1: float
        V_1 constant as calculated in `beam_constants`. This is described in the paper.
    K1g: np.ndarray (n_grid,)
        The weights grid when gridding the visibilities.
    K2gg: np.ndarray (n_grid,)
        The weights grid when gridding the squared visibility magnitudes.
    sigma_n: float
        The standard deviation of the complex noise.
    weights: np.ndarray (n_grid,)
        Weights to use when averaging estimates within a bin.

    Returns:
    --------
    Pk_sig_binned: (n_bins,)
        The standard error of the angular power spectrum estimator in the l-bins. 
    """
    n_bins = len(bin_edges) - 1
    Pk_sig_binned = np.zeros(n_bins)
    U = np.sqrt(nkxy[:, 0] ** 2 + nkxy[:, 1] ** 2)
    if weights is None:
        weights = np.ones(len(U))
    for i in range(n_bins):
        idx = np.where(
            (U >= bin_edges[i]) & (U < bin_edges[i + 1]) & (~np.isnan(E_g)) & (E_g > 0)
        )[0]
        if len(idx) == 0:
            Pk_sig_binned[i] = np.nan
        else:
            dU = nkxy[idx][:, None, :] - nkxy[idx][None, :, :]
            dU = np.sqrt(dU[:, :, 0] ** 2 + dU[:, :, 1] ** 2)
            w_2gg = (
                weights[idx][:, None]
                * weights[idx][None, :]
                * np.exp(-2 * (dU / sigma1) ** 2)
            )

            Pk_noise = (
                2
                * K2gg[idx]
                * np.exp(-((dU / sigma1) ** 2))
                * sigma_n**2
                / (K1g[idx][:, None] * K1g[idx][None, :])
            )

            Pk_sig_binned[i] = np.sqrt(
                np.sum(w_2gg * (E_g[idx] + Pk_noise) ** 2)
            ) / np.sum(weights[idx])

    return Pk_sig_binned


@njit(parallel=True)
def calc_vis(I, lm, uv, sigma_n, seed):
    """Calculate visibilities 
    
    Parameters:
    -----------
    I: np.ndarrray (n_pix,)
        Pixel values at all locations in the sky.
    lm: np.ndarray (n_pix, 2)
        lm-coordinates of all pixel values in the sky in units of projected radians.
    uv: np.ndarray (n_vis, 2)
        uv-coordinates of the baselines to calculate the visibilities at.
    sigma_n: float
        The standard deviation of the complex noise to add to the visibilities 
        in the same units as the sky pixels.
    seed: int
        Seed for the random number generator.

    Returns:
        vis: np.ndarray - complex (n_vis,)
    The visibilities at the given uv-coordinate locations.
    """
    np.random.seed(seed)
    N_vis = len(uv)
    l, m = lm.T
    u, v = uv.T
    vis = np.zeros(N_vis, dtype=np.complex128)
    for i in prange(N_vis):
        noise = sigma_n * (np.random.randn() + 1.0j * np.random.randn())
        vis[i] = np.sum(I * np.exp(-2j * np.pi * (u[i] * l + v[i] * m))) + noise

    return vis
