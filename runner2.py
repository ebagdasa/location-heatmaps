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
tot_eps = 1.0
level_sample_size = 100000
c = 0.1

for dropout in tqdm([0.0, 0.1, 0.2, 0.3, 0.4], leave=False):
    results[dropout] = dict()
    for dp_round_size in tqdm([5000, 6000, 7000, 8000, 9000, 10000],
                        leave=False):
        if (1-dropout)*sec_agg < dp_round_size:
            print(
                f'Skipping dropout round.')
            continue

        results[dropout][dp_round_size] = list()
        for j in tqdm(range(5), leave=False):
            print(f'using round: {dropout} {dp_round_size} {j}')
            if args.dp:
                eps_func = lambda x, num_regions: mechanisms.get_eps_from_two_std( c * np.sqrt(sec_agg / level_sample_size) * level_sample_size / num_regions)

                threshold_func = lambda i, p, eps, remaining: 2 / np.sqrt(sec_agg / level_sample_size) * mechanisms.get_std_from_eps(max(eps, remaining))
                total_epsilon_budget = tot_eps * level_sample_size
            else:
                eps_func = lambda x, y: None
                threshold_func = lambda i, prefix_len, eps, remaining: 5
                total_epsilon_budget = None

            res = run_experiment.run_experiment(true_image,
                           dataset,
                           level_sample_size=level_sample_size,
                           secagg_round_size=sec_agg,
                           threshold_func=threshold_func,
                           collapse_func=lambda threshold: 1/4 * threshold,
                           eps_func=eps_func,
                           total_epsilon_budget=total_epsilon_budget,
                           top_k=5,
                           partial=100,
                           max_levels=10,
                           total_size=1024,
                           min_dp_size=dp_round_size,
                           dropout_rate=dropout,
                           output_flag=False,
                           quantize=None,
                           save_gif=False,
                           positivity=False)

            results[dropout][dp_round_size].append(res[-1])

        torch.save(results, f'dropout_results_{args.name}.pt')