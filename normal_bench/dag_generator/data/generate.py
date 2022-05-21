from math import ceil
from random import sample
from functools import reduce
from collections import defaultdict

from .utils import draw


def random_levels(num_nodes):
    cpl = min(draw('cpl_dist.pkl', output_integer=True, path=[min(num_nodes, 35), ])[0], num_nodes)
    levels = draw('level_dist.pkl', num=num_nodes - cpl, output_integer=True, path=[min(cpl, 20), ])
    return levels + [*range(1, cpl + 1)]


def random_dag(num_nodes):
    if num_nodes == 1:
        return {0: []}

    # 随机生成level级别
    nodes = defaultdict(list)
    for n, l in enumerate(sorted(random_levels(num_nodes))):
        nodes[l].append(n)

    # 随机生成边
    parents = {n: [] for n in range(num_nodes)}
    for l in range(1, len(nodes)):
        for n in nodes[l]:
            for c in set(sample(nodes[l + 1], ceil(len(nodes[l + 1]) / len(nodes[l]) * 3 / 4))):
                parents[c].append(n)

    return parents


def random_job(n):
    job_dag = random_dag(n)
    task_info = [*zip(
        [f'T{k}' + reduce(str.__add__, [f'_{p}' for p in v], '') for k, v in job_dag.items()],
    )]

    return {
        'tasks': [','.join(map(str, info)) for info in task_info],
    }
