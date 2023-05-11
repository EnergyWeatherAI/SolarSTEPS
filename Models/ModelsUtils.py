import numpy as np
from pysteps.extrapolation.semilagrangian import extrapolate
from pysteps.noise.fftgenerators import initialize_nonparam_2d_fft_filter
from pysteps import utils as pyutils


def std_mask(x):
    return np.std(x).mean()


def get_lagrangian_space(images,
                         V):
    lagrangian_maps = []
    D = None
    for i in range(len(images) - 1):
        l_map, D = extrapolate(images[-(i + 1)],
                               V,
                               1,
                               return_displacement=True,
                               displacement_prev=D)
        lagrangian_maps.append(l_map[0])
    lagrangian_maps = lagrangian_maps[::-1]
    lagrangian_maps.append(images[-1])
    return np.array(lagrangian_maps)


def _get_mask(Size, idxi, idxj, win_fun):
    """Compute a mask of zeros with a window at a given position."""

    idxi = np.array(idxi).astype(int)
    idxj = np.array(idxj).astype(int)

    win_size = (idxi[1] - idxi[0], idxj[1] - idxj[0])
    if win_fun is not None:
        wind = pyutils.tapering.compute_window_function(win_size[0], win_size[1], win_fun)
        wind += 1e-6  # avoid zero values

    else:
        wind = np.ones(win_size)

    mask = np.zeros(Size)
    mask[idxi.item(0): idxi.item(1), idxj.item(0): idxj.item(1)] = wind

    return mask


def initialize_nonparam_2d_ssft_filter(field, **kwargs):
    """
    Function to compute the local Fourier filters using the Short-Space Fourier
    filtering approach.
    Parameters
    ----------
    field: array-like
        Two- or three-dimensional array containing one or more input fields.
        All values are required to be finite. If more than one field are passed,
        the average fourier filter is returned. It assumes that fields are stacked
        by the first axis: [nr_fields, y, x].
    Other Parameters
    ----------------
    win_size: int or two-element tuple of ints
        Size-length of the window to compute the SSFT (default (128, 128)).
    win_fun: {'hann', 'tukey', None}
        Optional tapering function to be applied to the input field, generated with
        :py:func:`pysteps.utils.tapering.compute_window_function`
        (default 'tukey').
    overlap: float [0,1[
        The proportion of overlap to be applied between successive windows
        (default 0.3).
    war_thr: float [0,1]
        Threshold for the minimum fraction of rain needed for computing the FFT
        (default 0.1).
    rm_rdisc: bool
        Whether or not to remove the rain/no-rain disconituity. It assumes no-rain
        pixels are assigned with lowest value.
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".
    Returns
    -------
    field: array-like
        Four-dimensional array containing the 2d fourier filters distributed over
        a 2d spatial grid.
        It can be passed to
        :py:func:`pysteps.noise.fftgenerators.generate_noise_2d_ssft_filter`.
    References
    ----------
    :cite:`NBSG2017`
    """

    if len(field.shape) < 2 or len(field.shape) > 3:
        raise ValueError("the input is not two- or three-dimensional array")
    if np.any(np.isnan(field)):
        raise ValueError("field must not contain NaNs")

    # defaults
    win_size = kwargs.get("win_size", (35, 35))

    if type(win_size) == int:
        win_size = (win_size, win_size)
    win_fun = kwargs.get("win_fun", "hann")
    overlap = kwargs.get("overlap", 0.9)
    war_thr = kwargs.get("war_thr", 0.01)
    rm_rdisc = kwargs.get("rm_disc", False)
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft_shape = field.shape if len(field.shape) == 2 else field.shape[1:]
        fft = pyutils.get_method(fft, shape=fft_shape)

    field = field.copy()

    # remove rain/no-rain discontinuity
    if rm_rdisc:
        field[field > field.min()] -= field[field > field.min()].min() - field.min()

    # dims
    if len(field.shape) == 2:
        field = field[None, :, :]
    nr_fields = field.shape[0]
    dim = field.shape[1:]
    dim_x = dim[1]
    dim_y = dim[0]

    # make sure non-rainy pixels are set to zero
    field -= field.min(axis=(1, 2))[:, None, None]

    # SSFT algorithm

    # prepare indices
    idxi = np.zeros(2, dtype=int)
    idxj = np.zeros(2, dtype=int)

    # number of windows
    num_windows_y = np.ceil(float(dim_y) / win_size[0]).astype(int)
    num_windows_x = np.ceil(float(dim_x) / win_size[1]).astype(int)

    # domain fourier filter
    F0 = initialize_nonparam_2d_fft_filter(
        field, win_fun=win_fun, donorm=True, use_full_fft=True, fft_method=fft
    )["field"]
    # and allocate it to the final grid
    F = np.zeros((num_windows_y, num_windows_x, F0.shape[0], F0.shape[1]))
    F += F0[np.newaxis, np.newaxis, :, :]

    # loop rows
    for i in range(F.shape[0]):
        # loop columns
        for j in range(F.shape[1]):

            # compute indices of local window
            idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0)))
            idxi[1] = int(
                np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_y))
            )
            idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0)))
            idxj[1] = int(
                np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_x))
            )

            # build localization mask
            # TODO: the 0.01 rain threshold must be improved
            mask = _get_mask(dim, idxi, idxj, win_fun)
            war = float(np.sum((field * mask[None, :, :]) > 0.01)) / (
                    (idxi[1] - idxi[0]) * (idxj[1] - idxj[0]) * nr_fields
            )

            if war > war_thr:
                # the new filter
                F[i, j, :, :] = initialize_nonparam_2d_fft_filter(
                    field * mask[None, :, :],
                    win_fun=None,
                    donorm=True,
                    use_full_fft=True,
                    fft_method=fft,
                )["field"]

    return {"field": F, "input_shape": field.shape[1:], "use_full_fft": True}


def generate_noise_2d_ssft_filter(F, std=1, randstate=None, seed=None, **kwargs):
    """
    Function to compute the locally correlated noise using a nested approach.
    Parameters
    ----------
    F: array-like
        A filter object returned by
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_ssft_filter` or
        :py:func:`pysteps.noise.fftgenerators.initialize_nonparam_2d_nested_filter`.
        The filter is a four-dimensional array containing the 2d fourier filters
        distributed over a 2d spatial grid.
    randstate: mtrand.RandomState
        Optional random generator to use. If set to None, use numpy.random.
    seed: int
        Value to set a seed for the generator. None will not set the seed.
    Other Parameters
    ----------------
    overlap: float
        Percentage overlap [0-1] between successive windows (default 0.2).
    win_fun: {'hann', 'tukey', None}
        Optional tapering function to be applied to the input field, generated with
        :py:func:`pysteps.utils.tapering.compute_window_function`
        (default 'tukey').
    fft_method: str or tuple
        A string or a (function,kwargs) tuple defining the FFT method to use
        (see "FFT methods" in :py:func:`pysteps.utils.interface.get_method`).
        Defaults to "numpy".
    Returns
    -------
    N: array-like
        A two-dimensional numpy array of non-stationary correlated noise.
    """
    input_shape = F["input_shape"]
    use_full_fft = F["use_full_fft"]
    F = F["field"]

    if len(F.shape) != 4:
        raise ValueError("the input is not four-dimensional array")
    if np.any(~np.isfinite(F)):
        raise ValueError("field contains non-finite values")

    if "domain" in kwargs.keys() and kwargs["domain"] == "spectral":
        raise NotImplementedError(
            "SSFT-based noise generator is not implemented in the spectral domain"
        )

    # defaults
    overlap = kwargs.get("overlap", 0.9)
    win_fun = kwargs.get("win_fun", "hann")
    fft = kwargs.get("fft_method", "numpy")
    if type(fft) == str:
        fft = pyutils.get_method(fft, shape=input_shape)

    if randstate is None:
        randstate = np.random

    # set the seed
    if seed is not None:
        randstate.seed(seed)

    dim_y = F.shape[2]
    dim_x = F.shape[3]
    dim = (dim_y, dim_x)

    # produce fields of white noise
    N = randstate.randn(dim_y, dim_x) * std
    fN = fft.fft2(N)

    # initialize variables
    cN = np.zeros(dim)
    sM = np.zeros(dim)

    idxi = np.zeros(2, dtype=int)
    idxj = np.zeros(2, dtype=int)

    # get the window size
    win_size = (float(dim_y) / F.shape[0], float(dim_x) / F.shape[1])

    # loop the windows and build composite image of correlated noise

    # loop rows
    for i in range(F.shape[0]):
        # loop columns
        for j in range(F.shape[1]):
            # apply fourier filtering with local filter
            lF = F[i, j, :, :]
            flN = fN * lF
            flN = np.array(fft.ifft2(flN).real)

            # compute indices of local window
            idxi[0] = int(np.max((i * win_size[0] - overlap * win_size[0], 0)))
            idxi[1] = int(
                np.min((idxi[0] + win_size[0] + overlap * win_size[0], dim_y))
            )
            idxj[0] = int(np.max((j * win_size[1] - overlap * win_size[1], 0)))
            idxj[1] = int(
                np.min((idxj[0] + win_size[1] + overlap * win_size[1], dim_x))
            )

            # build mask and add local noise field to the composite image
            M = _get_mask(dim, idxi, idxj, win_fun)
            cN += flN * M
            sM += M

    # normalize the field
    cN[sM > 0] /= sM[sM > 0]
    cN = (cN - cN.mean()) / cN.std()

    return cN

