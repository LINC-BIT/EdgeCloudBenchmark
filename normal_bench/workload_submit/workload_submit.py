import copy

import yaml
import random

from normal_bench.util import utils

from scipy import stats


class Submitter:
    def __init__(self, submit_model, jobs, time_interval):
        self.submit_model = submit_model
        self.jobs = jobs
        self.time_interval = time_interval

    def generate_workload_configuration(self, output_file, scheduler_name):
        model = self.submit_model.split(':')[0]
        param = None if len(self.submit_model.split(':')) < 2 else int(self.submit_model.split(":")[1])

        # 随机生成一批数据满足
        if model == 'normal':
            job_lists_real_answer = [[] for i in scheduler_name]
            for temp_job in self.jobs:
                for i in range(len(scheduler_name)):
                    for temp_task in temp_job:
                        temp_task.pod.spec.schedulerName = scheduler_name[i]
                    temp_gpu_job = [copy.deepcopy(i) for i in temp_job]
                    job_lists_real_answer[i].append(utils.to_dict(temp_gpu_job))
            for temp in range(len(output_file)):
                with open(output_file[temp], 'w') as f:
                    yaml.dump_all(job_lists_real_answer[temp], f)
        elif model == 'random':
            job_lists_real_answer = [[] for i in scheduler_name]
            random_jobs = random.shuffle(self.jobs)[:param]
            for temp_job in random_jobs:
                for i in range(len(scheduler_name)):
                    for temp_task in temp_job:
                        temp_task.pod.spec.schedulerName = scheduler_name[i]
                    temp_gpu_job = [copy.deepcopy(i) for i in temp_job]
                    job_lists_real_answer[i].append(utils.to_dict(temp_gpu_job))
            for temp in range(len(output_file)):
                with open(output_file[temp], 'w') as f:
                    yaml.dump_all(job_lists_real_answer[temp], f)
        elif model == 'average':
            # 重新生成任务
            random_jobs = []
            temp_index = 0
            job_lists_real_answer = [[] for i in scheduler_name]
            for i in range(self.time_interval):
                for j in range(param):
                    temp_job = copy.deepcopy(random.choice(self.jobs))
                    for temp_task in temp_job:
                        temp_task.startTime = temp_task.startTime + i * 1000
                        temp_task.pod.metadata.labels['job'] = f'job-{temp_index}'
                        temp_str = temp_task.pod.metadata.name.split('-')
                        temp_task.pod.metadata.name = f'job-{temp_index}-{temp_str[2]}-{temp_str[3]}'

                    random_jobs.append(temp_job)
                    temp_index += 1
            for temp_job in random_jobs:
                for i in range(len(scheduler_name)):
                    for temp_task in temp_job:
                        temp_task.pod.spec.schedulerName = scheduler_name[i]
                    temp_gpu_job = [copy.deepcopy(i) for i in temp_job]
                    job_lists_real_answer[i].append(utils.to_dict(temp_gpu_job))

            for temp in range(len(output_file)):
                with open(output_file[temp], 'w') as f:
                    yaml.dump_all(job_lists_real_answer[temp], f)
        else:
            # 重新生成任务
            random_jobs = []
            temp_index = 0
            job_lists_real_answer = [[] for i in scheduler_name]
            sample = stats.poisson.rvs(mu=param, size=1000 * self.time_interval)
            sample_index = 0
            submit_time = 0
            while submit_time < self.time_interval * 1000:
                temp_job = copy.deepcopy(random.choice(self.jobs))
                for temp_task in temp_job:
                    temp_task.startTime = temp_task.startTime + submit_time
                    temp_task.pod.metadata.labels['job'] = f'job-{temp_index}'
                    temp_str = temp_task.pod.metadata.name.split('-')
                    temp_task.pod.metadata.name = f'job-{temp_index}-{temp_str[2]}-{temp_str[3]}'
                random_jobs.append(temp_job)
                temp_index += 1
                submit_time += sample[sample_index]
                sample_index += 1
            for temp_job in random_jobs:
                for i in range(len(scheduler_name)):
                    for temp_task in temp_job:
                        temp_task.pod.spec.schedulerName = scheduler_name[i]
                    temp_gpu_job = [copy.deepcopy(i) for i in temp_job]
                    job_lists_real_answer[i].append(utils.to_dict(temp_gpu_job))

            for temp in range(len(output_file)):
                with open(output_file[temp], 'w') as f:
                    yaml.dump_all(job_lists_real_answer[temp], f)









