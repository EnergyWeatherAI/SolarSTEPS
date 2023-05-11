import numpy as np
from pysteps.extrapolation.semilagrangian import extrapolate
from pysteps.cascade.bandpass_filters import filter_gaussian
from pysteps.cascade.decomposition import decomposition_fft, recompose_fft
from pysteps.timeseries import autoregression, correlation
from pysteps.timeseries.autoregression import iterate_ar_model
from pysteps.nowcasts import utils as nowcast_utils
from pysteps.postprocessing.probmatching import nonparam_match_empirical_cdf
from multiprocessing.pool import Pool
from scipy.ndimage import generic_filter
from pysteps.utils.transformation import NQ_transform
from Models.ModelsUtils import get_lagrangian_space, initialize_nonparam_2d_ssft_filter, generate_noise_2d_ssft_filter
from functools import partial


def mean_mask(x):
    return np.nanmean(x)


def std_mask(x):
    return np.nanstd(x)


def generate_mask(img,
                  mask_func,
                  s):
    return generic_filter(img, mask_func, size=(s, s), mode='mirror')


class SolarSTEPS(object):
    def __init__(self,
                 ar_order,
                 n_cascade_levels,
                 probmatching=True,
                 norm=True,
                 local=True,
                 noise_coeff=None,
                 verbose=True,
                 noise_kwargs=None,
                 ar_kwargs=None,
                 norm_kwargs=None):

        self.ar_order = ar_order
        self.n_cascade_levels = n_cascade_levels
        self.probmatching = probmatching
        self.norm = norm
        self.verbose = verbose
        self.local = local
        self.noise_coeff = noise_coeff

        if ar_kwargs is None:
            ar_kwargs = dict()
        if noise_kwargs is None:
            noise_kwargs = dict()
        if norm_kwargs is None:
            norm_kwargs = dict()

        self.ar_win_type = ar_kwargs.get('ar_win_type', 'uniform')

        self.noise_win_size = noise_kwargs.get('noise_win_size', 90)
        self.noise_win_fun = noise_kwargs.get('noise_win_fun', 'hann')
        self.noise_std_win_size = noise_kwargs.get('noise_std_win_size', 15)
        self.noise_method = noise_kwargs.get('noise_method', 'local-SSFT')
        self.noise_overlap = noise_kwargs.get('noise_overlap', 0.9)

        self.extra_normalization = norm_kwargs.get('extra_normalization', True)

        if self.noise_coeff is None:
            self.noise_coeff = np.ones(n_cascade_levels)

    def initiate_forecast(self, input_maps, motion_field):
        input_maps = input_maps[-(self.ar_order + 1):]
        metadata = None
        x_shape, y_shape = input_maps.shape[1:]
        if self.norm:
            input_maps, metadata = NQ_transform(input_maps)

        # initialize noise
        if self.noise_method is not None:
            noise_dict = initialize_nonparam_2d_ssft_filter(input_maps[-1],
                                                            rm_rdisc=False,
                                                            win_fun=self.noise_win_fun,
                                                            win_size=self.noise_win_size,
                                                            overlap=self.noise_overlap,
                                                            seed=0)
        else:
            noise_dict = None

        # lagrangian space
        lag_maps = get_lagrangian_space(input_maps, motion_field)
        mask = np.isfinite(lag_maps)
        lag_maps[~mask] = 0

        # decomposition
        if self.n_cascade_levels > 1:
            filter_dict = filter_gaussian((x_shape, y_shape), n=self.n_cascade_levels)
            dec_lag_maps = []
            for i in range(lag_maps.shape[0]):
                map_ = lag_maps[i]
                mask_ = mask[i]
                decomposition_dict = decomposition_fft(map_, filter_dict, normalize=True, mask=mask_)
                dec_lag_maps.append(decomposition_dict['cascade_levels'])
            dec_lag_maps = np.array(dec_lag_maps)
        else:
            dec_lag_maps = lag_maps[:, np.newaxis, :, :]
            decomposition_dict = None
            filter_dict = None

        # autoregression coefficients
        if self.local:
            # retrieve AR win sizes, which corresponds to the scale of the different cascades
            central_wavenumbers = np.array([cf for cf in filter_dict["central_wavenumbers"]])
            scales = np.max((x_shape, y_shape)) / (2 * central_wavenumbers)
            scales = (np.ceil(scales) // 2) * 2 + 1
            ar_win_size = scales.astype(int)
            gamma = np.empty((self.n_cascade_levels, self.ar_order, x_shape, y_shape))
            phi = np.empty((self.n_cascade_levels, self.ar_order + 1, x_shape, y_shape))
            compute_phi = autoregression.estimate_ar_params_yw_localized

        else:
            gamma = np.empty((self.n_cascade_levels, self.ar_order))
            phi = np.empty((self.n_cascade_levels, self.ar_order + 1))
            compute_phi = autoregression.estimate_ar_params_yw
            ar_win_size = np.empty(self.n_cascade_levels) + np.inf

        if self.noise_method == 'local-SSFT':
            variance_mat = generic_filter(input_maps[-1],
                                          std_mask,
                                          size=(self.noise_std_win_size, self.noise_std_win_size),
                                          mode='mirror')

        elif self.noise_method == 'SSFT':
            variance_mat = 1

        else:
            raise Exception('The noise method must be in {static-local-SSFT, dynamic-local-SSFT, SSFT}')

        li_std = np.std(input_maps[-1])

        # compute gamma
        for i in range(self.n_cascade_levels):
            gamma[i, :] = correlation.temporal_autocorrelation(
                dec_lag_maps[:, i],
                mask=mask[0],
                window=self.ar_win_type,
                window_radius=ar_win_size[i]
            )

        if self.verbose and self.local is False:
            nowcast_utils.print_corrcoefs(gamma)

        if self.ar_order == 2:
            # adjust the lag-2 correlation coefficient to ensure that the AR(p)
            # process is stationary
            for i in range(self.n_cascade_levels):
                gamma[i, 1] = autoregression.adjust_lag2_corrcoef2(gamma[i, 0], gamma[i, 1])

        # estimate the parameters of the AR(p) model from the autocorrelation
        # coefficients and fill eventual nan with previous layers values
        for i in range(self.n_cascade_levels):
            phi[i, :] = compute_phi(gamma[i, :])
            phi_mask = np.isnan(phi[i, :])
            if i > 0:
                phi[i, :][phi_mask] = phi[i - 1, :][phi_mask]
            else:
                phi[i, :][phi_mask] = 0

        if self.verbose and self.local is False:
            nowcast_utils.print_ar_params(phi)

        return phi, dec_lag_maps, metadata, decomposition_dict, noise_dict, variance_mat, li_std

    def single_ens_forecast(self,
                            seed,
                            data):
        input_maps = data['input_maps']
        metadata = data['metadata']
        n_steps = data['n_steps']
        motion_field = data['motion_field']
        phi = data['phi']
        dec_lag_maps = data['dec_lag_maps']
        decomposition_dict = data['decomposition_dict']
        noise_dict = data['noise_dict']
        variance_mat = data['variance_mat']
        li_std = data['li_std']

        last_input = input_maps[-1]

        # generate noise
        if noise_dict is not None:
            noise = generate_noise_2d_ssft_filter(noise_dict,
                                                  std=variance_mat,
                                                  seed=seed,
                                                  fft_method='numpy',
                                                  win_fun=self.noise_win_fun,
                                                  win_size=self.noise_win_size,
                                                  overlap=self.noise_overlap,
                                                  )
            noise *= li_std

        else:
            noise = None

        # decomposition
        if noise is not None:
            if self.n_cascade_levels > 1:
                filter_dict = filter_gaussian(input_maps.shape[1:], n=self.n_cascade_levels)
                decomposed_noise_dict = decomposition_fft(noise, filter_dict, normalize=True)
                decomposed_noise = decomposed_noise_dict['cascade_levels']
            else:
                decomposed_noise = noise[np.newaxis, :, :]

        # forecast
        n = self.ar_order + 1
        new_dec_lag_maps = np.empty((n_steps + n,
                                     self.n_cascade_levels,
                                     last_input.shape[0],
                                     last_input.shape[1]))
        for i in range(self.n_cascade_levels):
            new_dec_lag_maps[:n, i, :, :] = dec_lag_maps[:, i, :, :]

        if self.n_cascade_levels > 1:
            yhat_dict = decomposition_dict
        displacement = None

        forecasted_maps = []
        for step in range(n_steps):
            yhat_lst = []
            for i in range(self.n_cascade_levels):
                if noise is not None:
                    eps = decomposed_noise[i] * self.noise_coeff[i]
                else:
                    eps = None
                yhat = iterate_ar_model(new_dec_lag_maps[step:step + n, i, :, :],
                                        phi[i],
                                        eps=eps)[-1]
                if self.extra_normalization:
                    yhat = yhat / np.std(yhat)
                new_dec_lag_maps[step + n, i, :, :] = yhat
                yhat_lst.append(yhat)

            yhat_arr = np.array(yhat_lst)
            if self.n_cascade_levels > 1:
                yhat_dict['cascade_levels'] = yhat_arr
                forecasted_map = recompose_fft(yhat_dict)
            else:
                forecasted_map = yhat_arr[0]

            if self.norm:
                forecasted_map, _ = NQ_transform(
                    forecasted_map,
                    metadata=metadata,
                    inverse=True)

            if self.probmatching:
                mask = ~np.isnan(forecasted_map)
                forecasted_map[mask] = nonparam_match_empirical_cdf(
                    forecasted_map[mask],
                    last_input[mask])

            forecasted_map, displacement = extrapolate(
                forecasted_map,
                motion_field,
                1,
                return_displacement=True,
                displacement_prev=displacement)

            forecasted_map = forecasted_map[0]
            forecasted_maps.append(forecasted_map)

        return np.array(forecasted_maps)

    def ensemble_forecast(self,
                          input_maps,
                          motion_field,
                          n_steps,
                          seeds):
        phi, dec_lag_maps, metadata, decomposition_dict, noise_dict, variance_mat, li_std = self.initiate_forecast(
            input_maps,
            motion_field)

        data = {'input_maps': input_maps,
                'motion_field': motion_field,
                'n_steps': n_steps,
                'phi': phi,
                'dec_lag_maps': dec_lag_maps,
                'metadata': metadata,
                'decomposition_dict': decomposition_dict,
                'noise_dict': noise_dict,
                'variance_mat': variance_mat,
                'li_std': li_std}

        partial_single_ens_forecast = partial(self.single_ens_forecast, data=data)

        with Pool() as pool:
            forecast_ensemble = pool.map(partial_single_ens_forecast, seeds)

        forecast_ensemble = np.array(forecast_ensemble)
        return forecast_ensemble
