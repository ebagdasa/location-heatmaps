from tqdm import tqdm
import numpy as np
import geo_utils
import run_experiment
import pickle
import torch
import mechanisms


import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--name', help='name')
# parser.add_argument('--dp', action='store_true')

args = parser.parse_args()

true_image, dataset = run_experiment.get_data('image.jpg')
total_dataset = np.load('total_dataset.npy')

split_dataset = geo_utils.makeGaussian(true_image, 1024, 500, [200, 900], convert=True, load=True)
res = torch.load('result5.pt')
threshold = 5
tree_prefix_list = res.tree_prefix_list
prefix_len = len(tree_prefix_list)
tree = res.tree
# noiser = mechanisms.GeometricNoise(10000, 1, eps)
# noiser = mechanisms.ZeroNoise()
total_size = 1024
partial=1000
dropout_rate=None
results = torch.load('results_jan25.pt')

for eps in [0.01, 0.1, 1.0]:
    for s in [10000000]:
        noiser = mechanisms.GeometricNoise(10000, 1, eps)
        samples = np.random.choice(split_dataset['total_dataset'], s, replace=False)
        result, grid_contour = geo_utils.make_step(samples, eps, threshold,
                                                   partial,
                                                   prefix_len, dropout_rate,
                                                   tree, tree_prefix_list,
                                                   noiser, quantize=None,
                                                   total_size=1024,
                                                   positivity=True)

        results.append([eps, s, result])
        torch.save(results, f'results_jan25.pt')