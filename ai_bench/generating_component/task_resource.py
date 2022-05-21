import pandas
import random
from ai_bench.util.utils import *

from sklearn import preprocessing
from sklearn.cluster import KMeans
import numpy as np

from ai_bench.generator.const import TASK_CLUSTER_VALUE, JOB_CLUSTER_VALUE
from ai_bench.data_generator.real_gpu_job import *
from ai_bench.util import utils


class TaskResource:
    def __init__(self, job_file_input, task_file_input, count_jobs, cloud_device):
        df = pandas.read_csv(job_file_input)
        df.fillna(0, inplace=True)  # 把NaN替换为0
        x = np.array(df.astype(float))

        x = preprocessing.minmax_scale(x)

        # 这个表示的是根据Plan_CPU、Plan_GPU、Plan_Mem、以及task_count进行分类的
        clf = KMeans(n_clusters=JOB_CLUSTER_VALUE)
        clf.fit(x)
        jobs_labels = clf.labels_  # 每个数据点所属分组

        res_series = pandas.Series(clf.labels_)
        lists = []
        sum_lists = 0
        df_jobs_labels = {'label_job': jobs_labels}
        data = pandas.DataFrame(df_jobs_labels)
        job_new_data = pandas.concat([df, data], axis=1)
        for i in range(0, JOB_CLUSTER_VALUE):
            temp_val = res_series[res_series.values == i].count()
            lists.append(temp_val)
            sum_lists += temp_val
        temp_count = count_jobs

        all_jobs = []
        for temp_list in lists[:-1]:
            temp_all_job_count = int(math.floor(temp_list / sum_lists * temp_count))
            all_jobs.append(temp_all_job_count)
            count_jobs -= temp_all_job_count
        all_jobs.append(count_jobs)

        tasks = pandas.read_csv(task_file_input)
        tasks.fillna(0, inplace=True)  # 把NaN替换为0

        tasks_x = np.array(tasks).astype(np.float64)[:, 0:3]
        tasks_x = preprocessing.minmax_scale(tasks_x)
        tasks_clf = KMeans(n_clusters=JOB_CLUSTER_VALUE)
        tasks_clf.fit(tasks_x)

        tasks_labels = tasks_clf.labels_  # 每个数据点所属分组

        df_tasks_labels = {'label_job': tasks_labels}
        tasks_data = pandas.DataFrame(df_tasks_labels)
        new_data = pandas.concat([tasks, tasks_data], axis=1)
        cols = [col for col in new_data]
        cols.append("label_task")
        answer_frame = pandas.DataFrame(columns=cols, dtype=np.float64)

        # 再次进行聚类
        for i in range(0, JOB_CLUSTER_VALUE):
            temp_frame = new_data[new_data.label_job == i]
            # 根据max_mem,avg_mem,avg_gpu,max_gpu,avg_cpu,max_cpu以及speed_time进行分类
            temp_x = np.array(temp_frame).astype(np.float64)[:, 4:11]
            temp_frame.reset_index()
            # 预处理
            temp_x = preprocessing.minmax_scale(temp_x)
            # 处理聚类的时候，要注意聚类的数量不要太大
            temp_clf = KMeans(n_clusters=min(TASK_CLUSTER_VALUE, len(temp_x)))
            temp_clf.fit(temp_x)

            labels = temp_clf.labels_  # 每个数据点所属分组

            c = {'label_task': labels}
            temp_data = pandas.DataFrame(data=c, index=temp_frame.index.tolist())

            temp_frame_answer = pandas.concat([temp_frame, temp_data], axis=1)

            answer_frame = pandas.concat([answer_frame, temp_frame_answer])

        # 开始抽选jobs
        job_id = 1
        job_lists = []
        for i in range(len(lists)):
            # 表示要抽取的数量
            jobs = all_jobs[i]
            if jobs > 0:
                temp_frame = job_new_data[job_new_data.label_job == i]
                while len(temp_frame) < jobs:
                    temp_frame = pandas.concat([temp_frame, temp_frame], ignore_index=True)
                temp_frame = temp_frame.sample(jobs).reset_index(drop=True)
                # 得出frame之后就要开始选择tasks了
                temp_label = tasks_clf.predict(
                    np.array(pandas.DataFrame([clf.cluster_centers_[i]]).astype(np.float64))[:, 1:])
                # print(temp_label[0])
                # 从中选取tasks
                tasks = answer_frame[answer_frame.label_job == temp_label[0]]

                for temp_index in range(len(temp_frame)):
                    # 按聚类结果挑选tasks
                    # print(temp_frame['task_count'][temp_index])
                    tasks_count = temp_frame['task_count'][temp_index]
                    tasks_counts = len(tasks)

                    temp_task_counts = tasks_count

                    cols = [col for col in tasks]
                    temp_answer_task_frame = pandas.DataFrame(columns=cols, dtype=np.float64)

                    for temp_task_index in range(TASK_CLUSTER_VALUE - 1):
                        temp_task_heap = tasks[tasks.label_task == temp_task_index]
                        task_counts = len(temp_task_heap)
                        tasks_new_count = int(math.floor(task_counts / tasks_counts * tasks_count))
                        if tasks_new_count > 0:
                            while len(temp_task_heap) < tasks_new_count:
                                temp_task_heap = pandas.concat([temp_task_heap, temp_task_heap], ignore_index=True)
                            real_tasks = temp_task_heap.sample(tasks_new_count)
                            temp_answer_task_frame = pandas.concat([temp_answer_task_frame, real_tasks])
                            temp_task_counts -= tasks_new_count
                    if temp_task_counts > 0:
                        temp_task_heap = tasks[tasks.label_task == TASK_CLUSTER_VALUE - 1]
                        while len(temp_task_heap) < temp_task_counts:
                            temp_task_heap = pandas.concat([temp_task_heap, temp_task_heap], ignore_index=True)
                        real_tasks = temp_task_heap.sample(temp_task_counts)
                        temp_answer_task_frame = pandas.concat([temp_answer_task_frame, real_tasks])

                    # 生成新的yaml文件
                    tasks_list = []
                    temp_answer_task_frame = temp_answer_task_frame.reset_index(drop=True)
                    # print(temp_answer_task_frame.head())

                    max_cpu = (round(float(temp_answer_task_frame['max_cpu'].mean()) / 100, 2) + (1 - random.random())) * cloud_device.cpu
                    max_mem = (round(float(temp_answer_task_frame['max_mem'].mean()), 2) + (1 - random.random())) * cloud_device.mem
                    avg_gpu = (round(float(temp_answer_task_frame['avg_cpu'].mean()) / 100, 2)) * cloud_device.gpu
                    avg_cpu = (round(float(temp_answer_task_frame['avg_cpu'].mean()) / 100, 2)) * cloud_device.cpu
                    avg_mem = (round(float(temp_answer_task_frame['avg_mem'].mean()), 2)) * cloud_device.mem
                    cpu_container = int(math.ceil(max_cpu))
                    mem_container = int(math.ceil(max_mem))
                    gpu_container = int(float(temp_answer_task_frame['max_gpu'].mean()) / 100 * cloud_device.gpu)

                    tasks_list.append(GPUTask(int(temp_answer_task_frame['inst_num'].mean()),
                                              Template(
                                                  TaskSpec(Container(
                                                      cpu_container,
                                                      mem_container,
                                                      gpu_container,
                                                      int(temp_answer_task_frame['time_speed'].mean()),
                                                      Resource(
                                                          max_cpu,
                                                          max_mem,
                                                          avg_cpu,
                                                          avg_mem,
                                                          avg_gpu
                                                      ), job_id, int(temp_answer_task_frame['inst_num'].mean()))),
                                                  int(temp_answer_task_frame['time_speed'].mean()) // 4,
                                                  job_id,
                                                  int(temp_answer_task_frame['inst_num'].mean())
                                              )))

                    temp_gpu_job = GPUJob(job_id, Spec(inst_num=1, tasks=tasks_list))
                    job_id += 1

                    # 随机生成一批数据满足
                    job_lists.append(utils.to_dict(temp_gpu_job))
        self.jobs_answer = job_lists

    def random_job(self, jobs_number):
        return self.jobs_answer
