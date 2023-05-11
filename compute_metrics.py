import numpy as np
import properscoring as ps
from pysteps.verification.spatialscores import fss
from pysteps.verification.detcatscores import det_cat_fct
from pysteps.verification.probscores import reldiag_init, reldiag_accum
from pysteps.postprocessing import ensemblestats


def compute_picp_pinaw(
        yhat_map,
        y_map,
        sample_size,
        ci=0.9):
    """
    :param yhat_map: (n_ens, m, n)
    :param y_map: (m, n)
    :param sample_size: int
    :param ci: float
    :return: float
    """
    nan_mask = ~np.isnan(yhat_map).any(axis=0)

    if sample_size == 'max':
        s = np.sum(nan_mask)
    else:
        s = sample_size
    sample_idx = np.random.choice(np.sum(nan_mask),
                                  replace=False,
                                  size=s)
    yhat_ = yhat_map[:, nan_mask][:, sample_idx]
    y_ = y_map[nan_mask][sample_idx]
    ub = np.nanquantile(yhat_, 1 - (1 - ci) / 2, axis=0)
    lb = np.nanquantile(yhat_, (1 - ci) / 2, axis=0)
    cond = (y_ <= ub) & (y_ >= lb)
    return np.sum(cond) / s, np.mean(ub - lb)


def compute_CRPS(yhat_map,
                 y_map):
    pred = yhat_map.reshape(yhat_map.shape[0],
                            yhat_map.shape[1],
                            -1).T

    obs = y_map.reshape(y_map.shape[0],
                        -1).T
    crps = ps.crps_ensemble(obs, pred).T.reshape(y_map.shape)
    return crps


def compute_fss(yhat_map,
                y_map,
                thresh,
                scale):
    return fss(yhat_map,
               y_map,
               thr=thresh,
               scale=scale)


def compute_CSI(yhat_map,
                y_map,
                thresh, ):
    return det_cat_fct(yhat_map,
                       y_map,
                       thr=thresh,
                       scores='CSI')['CSI']


def compute_rmse(yhat_map,
                 y_map):
    return np.sqrt(np.nanmean((yhat_map - y_map) ** 2))


def compute_bias(yhat_map,
                 y_map):
    diff = yhat_map - y_map
    return np.nanmean(diff), np.nanmax(diff), np.nanmin(diff)


def compute_ensemble_metrics(yhat,
                             y,
                             picp_sample_size=1000,
                             confidence_interval=0.9,
                             scale_lst=(1, 2, 4, 8, 16, 32, 64),
                             threshold_lst=(0.3, 0.6, 0.9)):
    result_dict = {}
    # PICP and PINAW

    picp_pinaw = [compute_picp_pinaw(yhat[:, j],
                                     y[j],
                                     sample_size=picp_sample_size,
                                     ci=confidence_interval) for j in range(len(y))]
    picp = np.array(picp_pinaw)[:, 0]
    pinaw = np.array(picp_pinaw)[:, 1]
    result_dict['picp'] = picp
    result_dict['pinaw'] = pinaw

    crps_maps = [compute_CRPS(yhat[:, j],
                              y[j]) for j in range(len(y))]
    result_dict['crps_map'] = crps_maps
    result_dict['avg_crps'] = np.nanmean(crps_maps, axis=(1, 2))

    rmse = [compute_rmse(np.nanmean(yhat[:, j], axis=0),
                         y[j]) for j in range(len(y))]
    bias = np.array([compute_bias(np.nanmean(yhat[:, j], axis=0),
                                  y[j]) for j in range(len(y))])
    result_dict['rmse'] = np.array(rmse)
    result_dict['avg_bias'] = bias[:, 0]
    result_dict['max_bias'] = bias[:, 1]
    result_dict['min_bias'] = bias[:, 2]

    csi_dict = {}
    fss_dict = {}
    for t in threshold_lst:
        csi = np.array([compute_CSI(np.nanmean(yhat[:, j], axis=0),
                                    y[j],
                                    t) for j in range(len(y))])
        csi_dict[t] = csi
        fss_dict[t] = {}
        for scale in scale_lst:
            fs_score = np.array([compute_fss(np.nanmean(yhat[:, j], axis=0),
                                             y[j],
                                             t,
                                             scale) for j in range(len(y))])
            fss_dict[t][scale] = fs_score
    result_dict['csi'] = csi_dict
    result_dict['fss'] = fss_dict
    return result_dict


def compute_det_metrics(yhat,
                        y,
                        scale_lst=(1, 2, 4, 8, 16, 32, 64),
                        threshold_lst=(0.3, 0.6, 0.9)):
    result_dict = {}
    rmse = [compute_rmse(yhat[j],
                         y[j]) for j in range(len(y))]
    bias = np.array([compute_bias(yhat[j],
                                  y[j]) for j in range(len(y))])
    result_dict['rmse'] = np.array(rmse)
    result_dict['avg_bias'] = bias[:, 0]
    result_dict['max_bias'] = bias[:, 1]
    result_dict['min_bias'] = bias[:, 2]

    csi_dict = {}
    fss_dict = {}
    for t in threshold_lst:
        csi = np.array([compute_CSI(yhat[j],
                                    y[j],
                                    t) for j in range(len(y))])
        csi_dict[t] = csi
        fss_dict[t] = {}
        for scale in scale_lst:
            fs_score = np.array([compute_fss(yhat[j],
                                             y[j],
                                             t,
                                             scale) for j in range(len(y))])
            fss_dict[t][scale] = fs_score
    result_dict['csi'] = csi_dict
    result_dict['fss'] = fss_dict
    return result_dict


def init_reldiagrams(thresh_lst):
    reldiag_dict = {}
    for t in thresh_lst:
        reldiag_dict[t] = reldiag_init(t)
    return reldiag_dict


def accum_reldiagrams(yhat,
                      y,
                      reldiag_dict):
    for t in reldiag_dict:
        prob = ensemblestats.excprob(yhat, t, ignore_nan=True)
        reldiag_accum(reldiag_dict[t], prob, y)
