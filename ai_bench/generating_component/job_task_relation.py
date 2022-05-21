import pandas
import random
from ai_bench.util.utils import *


# JobTaskRelation 记录每一个job所含的task数量
class JobTaskRelation:
    def __init__(self, input_job_file):
        df = pandas.read_csv(input_job_file)
        df.fillna(0, inplace=True)  # 把NaN替换为0

        # job_map 记录每一个job所含task的数量
        job_map = {}
        for i in range(len(df)):
            task_number = int(df['task_count'][i])
            if job_map.get(task_number) is None:
                job_map[task_number] = 0
            job_map[task_number] += 1
        count = 0
        key_set = []
        value_set = []
        for key, value in job_map.items():
            count += value
            key_set.append(key)
            value_set.append(value)
        for i in range(len(value_set)):
            value_set[i] = value_set[i] / count
        for i in range(1, len(value_set)):
            value_set[i] += value_set[i - 1]
        self.key_set = key_set
        self.value_set = value_set

    def random_task_count(self):
        random_value = random.random()
        # 用二分进行优化
        index = binary_search(self.value_set, random_value)
        return self.key_set[index]





