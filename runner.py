from tqdm import tqdm
import numpy as np
import geo_utils
import run_experiment
import pickle
import torch
import mechanisms

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name', help='name')
parser.add_argument('--dp', action='store_true')

args = parser.parse_args()

true_image, dataset = run_experiment.get_data('image.jpg')
total_dataset = np.load('total_dataset.npy')


results = list()


for level_sample_size in [10000, 100000, 1000000]:
    if args.dp:
        eps_func = lambda x, y: mechanisms.get_eps_var(1 / 10 * level_sample_size / y)
        threshold_func = lambda i, prefix_len, eps, remaining: mechanisms.std_geom(
            max(eps, remaining))
        total_epsilon_budget = 1 * level_sample_size
    else:
        eps_func = lambda x, y: None
        threshold_func = lambda i, prefix_len, eps, remaining: 5
        total_epsilon_budget = None


    res = run_experiment.run_experiment(true_image,
                   total_dataset,
                   level_sample_size=level_sample_size,
                   secagg_round_size=10000,
                   threshold_func=threshold_func,
                   collapse_threshold=None,
                   eps_func=eps_func,
                   total_epsilon_budget=total_epsilon_budget,
                   top_k=5,
                   partial=100,
                   max_levels=10,
#                    threshold_func=lambda i, prefix_len, eps, left_budget: 2*mechanisms.std_geom(eps, 1),
                   collapse_func=None,
                   total_size=1024,
                   min_dp_size=None,
                   dropout_rate=None,
                   output_flag=False,
                   quantize=None,
                   save_gif=False,
                       positivity=True)

    results.append(res)
    torch.save(results, f'results_{args.name}.pt')