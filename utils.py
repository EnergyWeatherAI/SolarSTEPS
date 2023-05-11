import pickle as pkl
import numpy as np
import yaml
from pysteps.extrapolation.semilagrangian import extrapolate
import properscoring as ps
from pysteps.verification.ensscores import ensemble_skill
from pysteps.postprocessing import ensemblestats
from pysteps.verification.probscores import reldiag, reldiag_init, reldiag_accum

ROOT_PATH = '/Users/cea3/Desktop/Projects/OpticalFlowSSR/'


def open_pkl(path: str):
    with open(path, 'rb') as o:
        pkl_file = pkl.load(o)
    return pkl_file


def open_yaml(path: str):
    with open(path) as o:
        yaml_file = yaml.load(o, Loader=yaml.FullLoader)
    return yaml_file


def get_sat_idx(ts_lst,
                df):
    return df[df.timestamps.isin(ts_lst)].sat_idx.values.astype(int)


def get_data(subset='train'):
    path = str(open_yaml(ROOT_PATH + 'config.yml')['data_path'])
    dataset = open_pkl(path)
    ki_images = dataset['ki_images']
    df = dataset['{}_df'.format(subset)]

    input_ts = np.array(dataset['in_{}_ts'.format(subset)])
    output_ts = np.array(dataset['out_{}_ts'.format(subset)])

    input_idx = np.array(dataset['in_{}_idx'.format(subset)])
    output_idx = np.array(dataset['out_{}_idx'.format(subset)])

    return ki_images, df, input_ts, output_ts, input_idx, output_idx


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


