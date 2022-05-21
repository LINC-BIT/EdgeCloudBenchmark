from ai_bench.data_generator.gpu_job import *
from collections import Iterable
import yaml


def to_dict(obj):
    data = {}
    if isinstance(obj, Iterable):
        data = []
        for temp_obj in obj:
            temp_data = {}
            for key, value in temp_obj.__dict__.items():
                try:
                    temp_data[key] = to_dict(value)
                except AttributeError:
                    temp_data[key] = value
            data.append(temp_data)
    else:
        for key, value in obj.__dict__.items():
            try:
                data[key] = to_dict(value)
            except AttributeError:
               data[key] = value
    return data


if __name__ == '__main__':
    gpu_jobs = GPUJob(1, Spec(1, [GPUTask(1, 1, TaskSpec(Container(1, 750, 2, 39142,
                                                                   Resource(0.57, 1500.0, 0.29, 750.0))))]))
    print(to_dict(gpu_jobs))
    file = open('./test.yaml', 'w', encoding='utf-8')
    yaml.dump(to_dict(gpu_jobs), file)
    file.close()
