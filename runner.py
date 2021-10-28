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
TOPK=5
TOTAL_SIZE = 1024

results = dict()
sec_agg = 10000
tail_coefficient = 1/10
# [10000, 50000, 100000, 200000, 500000, 800000, 1000000 ,2000000, 5000000, 10000000]
# 0.5, 1, 2
counts = list()
tot_eps = 2.0
level_sample_size = 100000

users = 1000000
secagg_size = 10000
c = 1/10


results = run_experiment.run_experiment(true_image,
                   dataset,
                   level_sample_size=users,
                   secagg_round_size=secagg_size,
                   eps_func=lambda x, num_regions: mechanisms.get_eps_from_two_std(c * np.sqrt(secagg_size / users) * users/num_regions),
                   threshold_func=lambda i, prefix_len, eps, remaining: 2 / np.sqrt(c) * mechanisms.get_std_from_eps( eps),
                   collapse_func=lambda threshold: max(5, 1/4 * threshold),
                   total_epsilon_budget=tot_eps*users,
                   top_k=TOPK,
                   partial=1000,
                   max_levels=10,
                   total_size=TOTAL_SIZE,
                   # min_dp_size=9000,
                   # dropout_rate=0.1,
                   output_flag=True,
                   quantize=None,
                   save_gif=False,
                   positivity=False, start_with_level=4,
                   last_result_ci=False)

name = 'results_1mln_2eps_3.pt'
print(name)
torch.save(results, name)