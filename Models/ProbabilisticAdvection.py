from pysteps import nowcasts
import numpy as np
from multiprocessing import Pool


class ProbabilisticAdvection(object):

    def __init__(self,
                 ens_members=25,
                 alpha=0,
                 beta=0,
                 return_motion_field=False,
                 angle_pert_dist='normal'
                 ):
        self.ens_members = ens_members
        self.extrapolate = nowcasts.get_method('extrapolation')
        self.alpha = alpha
        self.beta = beta
        self.return_motion_field = return_motion_field
        self.angle_pert_dist = angle_pert_dist

    def ens_forecast(self,
                     seed):
        np.random.seed(seed)

        x = self.motion_field[0]
        y = self.motion_field[1]

        motion_abs = x ** 2 + y ** 2

        alpha_noise = np.random.normal(0,
                                       self.alpha,
                                       self.last_map.shape)
        abs_field = motion_abs + alpha_noise
        mask = abs_field < 0
        flag = mask.any()
        abs_field[mask] = motion_abs[mask]
        s = 0
        while flag and s < 10:
            alpha_noise = np.random.normal(0,
                                           self.alpha,
                                           np.sum(mask))
            abs_field[mask] = motion_abs[mask] + alpha_noise
            mask = abs_field < 0
            flag = mask.any()
            abs_field[mask] = motion_abs[mask]
            s += 1
        abs_field[abs_field < 0] = 0
        pert_abs = np.sqrt(abs_field)

        # angle perturbation
        pert_motion_field = np.empty(self.motion_field.shape)

        if self.angle_pert_dist == 'normal':
            beta_noise = np.random.normal(0,
                                          self.beta,
                                          self.last_map.shape)
        elif self.angle_pert_dist == 'vonmises':
            beta_noise = np.random.vonmises(0,
                                            self.beta,
                                            self.last_map.shape)
        else:
            raise Exception('angle_pert_dist must be either normal or vonmises')

        pert_motion_field[0, :, :] = pert_abs * np.cos(np.arctan(y / x) + beta_noise)
        pert_motion_field[1, :, :] = pert_abs * np.sin(np.arctan(y / x) + beta_noise)

        yhat_maps = self.extrapolate(self.last_map,
                                     pert_motion_field,
                                     self.n_steps)
        if self.return_motion_field:
            return yhat_maps, pert_motion_field
        else:
            return yhat_maps

    def maps_forecast(self,
                      n_steps,
                      input_maps,
                      motion_field):

        self.n_steps = n_steps
        self.motion_field = motion_field
        self.last_map = input_maps[-1]

        if self.ens_members > 1:
            with Pool() as p:
                yhat_maps = p.map(self.ens_forecast,
                                  np.arange(self.ens_members))
        else:
            yhat_maps = self.extrapolate(self.last_map,
                                         self.motion_field,
                                         n_steps)
        if self.return_motion_field:
            return yhat_maps
        else:
            return np.array(yhat_maps)
