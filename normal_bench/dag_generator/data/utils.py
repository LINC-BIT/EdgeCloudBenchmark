import math
import pickle
from pathlib import Path
from collections import defaultdict


DATA_DIR = Path(__file__).resolve().parents[0] / 'distributions'
DIST_CACHE = {}
SAMPLE_CACHE = defaultdict(list)


def draw(dist_name, num=1, path=[], output_integer=True):
    # 从给定分布中选取出一个随机分布
    if dist_name not in DIST_CACHE:
        with (DATA_DIR / dist_name).open('rb') as f:
            DIST_CACHE[dist_name] = pickle.load(f)

    dist = DIST_CACHE[dist_name]
    for p in path:
        dist = dist[p]

    # 从缓存中获取结果，因为关键路径和level的生成耗费时间
    cache_name = dist_name + ''.join(map(str, path))
    cache = SAMPLE_CACHE[cache_name]
    while len(cache) < num:
        cache += list(dist.rvs(size=10240))

    samples = cache[:num]
    if output_integer:
        samples = [*map(math.ceil, samples)]

    SAMPLE_CACHE[cache_name] = cache[num:]

    return samples
