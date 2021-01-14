from tqdm import tqdm
import numpy as np
import geo_utils
import run_experiment
import pickle
import torch
import mechanisms
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--name', help='name')
parser.add_argument('--dp', action='store_true')

args = parser.parse_args()

true_image, dataset = run_experiment.get_data('image.jpg')


results = dict()
sec_agg = 10000
tail_coefficient = 1/10
# [10000, 50000, 100000, 200000, 500000, 800000, 1000000 ,2000000, 5000000, 10000000]
# 0.5, 1, 2
counts = list()

for tot_eps in tqdm([0.01, 0.1, 1, 10], leave=True):
    results_eps = defaultdict(list)
    for i, level_sample_size in tqdm(enumerate([500000, 800000, 1000000]), leave=True):
        print(f'Level sample size: {level_sample_size}. Epsilon: {tot_eps}.')
        for j in tqdm(range(5), leave=False):
            c = np.sqrt(sec_agg / level_sample_size)

            if args.dp:
                eps_func = lambda x, num_regions: mechanisms.get_eps_from_two_std(1 / 10 * np.sqrt(c) * level_sample_size / num_regions)

                threshold_func = lambda i, p, eps, remaining: 2 / np.sqrt(
                    c) * mechanisms.get_std_from_eps(max(eps, remaining))
                total_epsilon_budget = tot_eps * level_sample_size
            else:
                eps_func = lambda x, y: None
                threshold_func = lambda i, prefix_len, eps, remaining: 5
                total_epsilon_budget = None


            res = run_experiment.run_experiment(true_image,
                           dataset,
                           level_sample_size=level_sample_size,
                           secagg_round_size=10000,
                           threshold_func=threshold_func,
                           collapse_threshold=None,
                           eps_func=eps_func,
                           total_epsilon_budget=total_epsilon_budget,
                           top_k=5,
                           partial=100,
                           max_levels=10,
                           collapse_func=None,
                           total_size=1024,
                           min_dp_size=None,
                           dropout_rate=None,
                           output_flag=False,
                           quantize=None,
                           save_gif=False,
                           positivity=False,
                           start_with_level=0)
            results_eps[level_sample_size].append(res[-1])

    results[tot_eps] = results_eps
    torch.save(results, f'results_{args.name}.pt')