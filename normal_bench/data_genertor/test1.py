from collections import Iterable
import yaml
from normal_bench.data_genertor.cpu_pod import *


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
    cpu_jobs = [CPUPod(Pod(MetaData(1, 2, 1, 'cpu', 'cloud'),
                           TaskSpec(Container(2, 330, 2000, Resource(0.5, 530, 0.42, 30),
                                              'edge0', 'default-scheduler'))), 0),
                CPUPod(Pod(MetaData(2, 2, 2, 'cpu', 'cloud'),
                           TaskSpec(Container(2, 330, 2000, Resource(0.6, 520, 0.41, 10),
                                              'edge1', 'default-scheduler'))), 0)]

    cpu_jobs = to_dict(cpu_jobs)
    print(cpu_jobs)
    file = open('./test.yaml', 'w', encoding='utf-8')
    yaml.dump(cpu_jobs, file)
    file.close()
