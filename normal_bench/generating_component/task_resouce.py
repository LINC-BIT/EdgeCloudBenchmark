import yaml

from normal_bench.dag_generator.data import generate
import pandas

import numpy as np

import copy
from sklearn import preprocessing

import random

from sklearn.cluster import KMeans

from normal_bench.generator.const import JOB_CLUSTER, TASK_CLUSTER, MAX_LEVEL_EDGE_CLOUD_EDGE

import math

from normal_bench.data_genertor.cpu_pod import *
from normal_bench.util import utils

class DAGTask:
    def __init__(self, node, level):
        self.node = node
        self.level = level


class TaskResource:
    def __init__(self, cpu_max, mem_max, cpu_avg, mem_avg, duration):
        self.cpu_max = round(cpu_max, 2)
        self.mem_max = round(mem_max * 1024, 2)
        self.cpu_avg = round(cpu_avg, 2)
        self.mem_avg = round(mem_avg * 1024, 2)
        self.duration = duration


def reorder(dag_task):
    return dag_task.level


def reorder2(task_resource):
    return task_resource.cpu_avg


def find_max_level(total_node, job_type):
    level_min = 0
    if job_type == 'edge-cloud' or job_type == 'cloud-edge':
        level_min = 2
    else:
        level_min = 3

    edges = [[] for i in range(total_node)]
    # 标记level
    edges_level = [0 for i in range(total_node)]

    tasks_dag = []

    while max(edges_level) < level_min:
        edges = [[] for i in range(total_node)]
        # 标记level
        edges_level = [0 for i in range(total_node)]
        tasks_dag = generate.random_job(total_node)['tasks']
        for temp_task in tasks_dag:
            temp_task_str = temp_task.split('_')
            # total_edge += len(temp_task_str) - 1
            answer = []
            node = int(temp_task_str[0][1:])
            if len(temp_task_str) > 1:
                for ttt in temp_task_str[1:]:
                    answer.append(int(ttt))
            else:
                edges_level[node] = 1
            edges[node] = answer
        # level
        for i in range(total_node):
            for j in range(total_node):
                if len(edges[j]) != 0:
                    for temp_edge in edges[j]:
                        if edges_level[temp_edge] != 0:
                            edges_level[j] = max(edges_level[j], edges_level[temp_edge] + 1)

        # edges_level[j]是按层级进行排序的
    tasks_dag_new_data = []
    for i in range(total_node):
        tasks_dag_new_data.append(DAGTask(tasks_dag[i], edges_level[i]))
    tasks_dag_new_data.sort(key=reorder)
    return tasks_dag_new_data


scheduler_name = ['linc-scheduler-mrp', 'linc-scheduler-bra', 'linc-scheduler-lrp', 'linc-scheduler-ep']


class NormalTaskResource:
    def __init__(self, file_name_job, file_name_tasks, count_jobs, scene, task_percentage, edge_device, cloud_device):
        df = pandas.read_csv(file_name_job)
        df.fillna(0, inplace=True)  # 把NaN替换为0
        x = np.array(df.astype(float))

        x = preprocessing.minmax_scale(x)
        # 根据plan_cpu、plan_mem、task_count进行分类
        clf = KMeans(n_clusters=TASK_CLUSTER)
        clf.fit(x)
        jobs_labels = clf.labels_  # 每个数据点所属分组

        res_series = pandas.Series(clf.labels_)

        lists = []
        sum_lists = 0
        df_jobs_labels = {'label_job': jobs_labels}
        data = pandas.DataFrame(df_jobs_labels)
        job_new_data = pandas.concat([df, data], axis=1)

        for i in range(0, JOB_CLUSTER):
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

        # 对task进行分析
        # tasks = pandas.read_csv(file_name_tasks)
        tasks = pandas.read_csv(file_name_tasks)
        tasks.fillna(0, inplace=True)  # 把NaN替换为0

        tasks_x = np.array(tasks).astype(np.float64)[:, 2:4]
        tasks_clf = KMeans(n_clusters=JOB_CLUSTER)
        tasks_clf.fit(tasks_x)

        tasks_series = pandas.Series(tasks_clf.labels_)
        tasks_centers = tasks_clf.cluster_centers_  # 两组数据点的中心点
        tasks_labels = tasks_clf.labels_  # 每个数据点所属分组

        df_tasks_labels = {'label_job': tasks_labels}
        tasks_data = pandas.DataFrame(df_tasks_labels)
        new_data = pandas.concat([tasks, tasks_data], axis=1)
        cols = [col for col in new_data]
        cols.append("label_task")
        answer_frame = pandas.DataFrame(columns=cols, dtype=np.float64)

        # 再次进行聚类
        for i in range(0, JOB_CLUSTER):
            temp_frame = new_data[new_data.label_job == i]
            # 根据max_mem,avg_mem,avg_gpu,max_gpu,avg_cpu,max_cpu以及speed_time进行分类
            temp_x = np.array(temp_frame).astype(np.float64)[:, 4:9]
            temp_frame.reset_index()
            # 预处理
            temp_x = preprocessing.minmax_scale(temp_x)
            temp_clf = KMeans(n_clusters=min(TASK_CLUSTER, len(temp_x)))
            temp_clf.fit(temp_x)

            labels = temp_clf.labels_  # 每个数据点所属分组

            c = {'label_task': labels}
            temp_data = pandas.DataFrame(data=c, index=temp_frame.index.tolist())

            temp_frame_answer = pandas.concat([temp_frame, temp_data], axis=1)

            answer_frame = pandas.concat([answer_frame, temp_frame_answer])

        # 开始抽选jobs
        job_id = 1
        job_lists = []

        job_lists_real = [[] for i in scheduler_name]
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
                # 从中选取tasks
                tasks = answer_frame[answer_frame.label_job == temp_label[0]]

                for temp_index in range(len(temp_frame)):
                    # 按聚类结果挑选tasks
                    tasks_count = int(temp_frame['task_count'][temp_index])
                    # 根据边生成符合的DAG依赖
                    tasks_counts = len(tasks)

                    temp_task_counts = tasks_count

                    cols = [col for col in tasks]
                    temp_answer_task_frame = pandas.DataFrame(columns=cols, dtype=np.float64)

                    for temp_task_index in range(TASK_CLUSTER - 1):
                        temp_task_heap = tasks[tasks.label_task == temp_task_index]
                        task_counts = len(temp_task_heap)
                        tasks_new_count = int(math.ceil(task_counts / tasks_counts * tasks_count))
                        if tasks_new_count > 0:
                            while len(temp_task_heap) < tasks_new_count:
                                temp_task_heap = pandas.concat([temp_task_heap, temp_task_heap], ignore_index=True)
                            real_tasks = temp_task_heap.sample(tasks_new_count)
                            temp_answer_task_frame = pandas.concat([temp_answer_task_frame, real_tasks])
                            temp_task_counts -= tasks_new_count
                    if temp_task_counts > 0:
                        temp_task_heap = tasks[tasks.label_task == TASK_CLUSTER - 1]
                        while len(temp_task_heap) < temp_task_counts:
                            temp_task_heap = pandas.concat([temp_task_heap, temp_task_heap], ignore_index=True)
                        real_tasks = temp_task_heap.sample(temp_task_counts)
                        temp_answer_task_frame = pandas.concat([temp_answer_task_frame, real_tasks])
                    # 生成新的yaml文件
                    tasks_list = []
                    temp_answer_task_frame = temp_answer_task_frame.reset_index(drop=True)

                    temp_answer_task_frame = temp_answer_task_frame.sort_values(by=['cpu_avg', 'mem_avg'],
                                                                                ascending=True).reset_index()

                    # 将dag依赖按照层级的方式进行排序
                    tasks_dag_new_data = find_max_level(tasks_count, scene)

                    # for temp_tasks_dag in tasks_dag_new_data:
                    #     print(f'{temp_tasks_dag.node}------{temp_tasks_dag.level}')

                    # 根据level层次来选择任务
                    edge_tasks = []
                    cloud_tasks = []

                    for temp_real_task_index in range(len(temp_answer_task_frame)):
                        if temp_real_task_index < len(temp_answer_task_frame) // 2:
                            edge_tasks.append(TaskResource(
                                float(temp_answer_task_frame['cpu_max'][temp_real_task_index]) / 30,
                                float(temp_answer_task_frame['mem_max'][temp_real_task_index]),
                                float(temp_answer_task_frame['cpu_avg'][temp_real_task_index]) / 30,
                                float(temp_answer_task_frame['mem_avg'][temp_real_task_index]),
                                int(temp_answer_task_frame['duration'][temp_real_task_index])))
                        else:
                            cloud_tasks.append(TaskResource(
                                float(temp_answer_task_frame['cpu_max'][temp_real_task_index]) / 30,
                                float(temp_answer_task_frame['mem_max'][temp_real_task_index]),
                                float(temp_answer_task_frame['cpu_avg'][temp_real_task_index]) / 30,
                                float(temp_answer_task_frame['mem_avg'][temp_real_task_index]),
                                int(temp_answer_task_frame['duration'][temp_real_task_index])))

                    random.shuffle(edge_tasks)
                    random.shuffle(cloud_tasks)
                    tasks_list = [CPUPod(Pod(MetaData(job_id, tasks_count, 0, 'cpu', 'default'),
                                             TaskSpec(
                                                 Container(0,
                                                           0,
                                                           0,  # 毫秒
                                                           Resource(0,
                                                                    0,
                                                                    0,
                                                                    0)),
                                                 'edge1',
                                                 'default-scheduler'
                                             )), 0) for i in
                                  range(len(tasks_dag_new_data))]
                    tasks_list2 = [CPUPod(Pod(MetaData(job_id, tasks_count, 0, 'cpu', 'default'),
                                              TaskSpec(
                                                  Container(0,
                                                            0,
                                                            0,  # 毫秒
                                                            Resource(0,
                                                                     0,
                                                                     0,
                                                                     0)),
                                                  'edge1',
                                                  'default-scheduler'
                                              )), 0) for i in
                                   range(len(tasks_dag_new_data))]

                    # 根据层级来安排云边协同任务
                    # scene1

                    new_tasks_not_real = []

                    if scene == 'edge-cloud':
                        # 这里设置edge负载的个数
                        half_index = int(math.floor(len(tasks_dag_new_data) * task_percentage))
                        for temp_real_task_index in range(len(tasks_dag_new_data)):
                            dag_task = tasks_dag_new_data[temp_real_task_index].node.split('_')
                            node_index = int(dag_task[0][1:])
                            if len(dag_task) > 0:
                                dag_task = [int(t) for t in dag_task[1:]]
                            start_time = 0
                            start_time2 = 0
                            for t in dag_task:
                                start_time = max(start_time, tasks_list[t].startTime + int(
                                    tasks_list[t].pod.spec.containers[0].args[5]))
                                start_time2 = max(start_time2, tasks_list2[t].startTime + int(
                                    tasks_list2[t].pod.spec.containers[0].args[5]))

                            # edge_cpu_max = round((edge_tasks[temp_real_task_index].cpu_max + (1 - random.random())) * edge_device.cpu, 2)
                            # cloud_cpu_max = round((edge_tasks[temp_real_task_index].cpu_max + (1 - random.random())) * cloud_device.cpu, 2)
                            # edge_mem_max = edge_tasks[temp_real_task_index].mem_max * edge_device.mem
                            # cloud_mem_max = edge_tasks[temp_real_task_index].mem_max * cloud_device.mem
                            # edge_cpu_avg = edge_tasks[temp_real_task_index].cpu_avg * edge_device.mem
                            # cloud_cpu_avg = edge_tasks[temp_real_task_index].cpu_avg * cloud_device.mem
                            #
                            # edge_mem_avg = edge_tasks[temp_real_task_index].mem_avg * edge_device.mem
                            # cloud_mem_avg = edge_tasks[temp_real_task_index].mem_avg * cloud_device.mem
                            #
                            # cpu_container = int(math.ceil(edge_cpu_max))
                            # mem_container = int(math.ceil(edge_mem_max))

                            if temp_real_task_index < half_index:
                                edge_cpu_max = round((edge_tasks[temp_real_task_index].cpu_max + (
                                            1 - random.random())) * edge_device.cpu, 2)
                                edge_mem_max = edge_tasks[temp_real_task_index].mem_max * edge_device.mem
                                edge_cpu_avg = edge_tasks[temp_real_task_index].cpu_avg * edge_device.mem
                                edge_mem_avg = edge_tasks[temp_real_task_index].mem_avg * edge_device.mem
                                cpu_container = int(math.ceil(edge_cpu_max))
                                mem_container = int(math.ceil(edge_mem_max))

                                tasks_list[node_index] = CPUPod(
                                    Pod(MetaData(job_id, tasks_count, temp_real_task_index, 'cpu', 'edge1'),
                                        TaskSpec(
                                            Container(cpu_container,
                                                      mem_container,
                                                      int(edge_tasks[temp_real_task_index].duration) * 1000,  # 毫秒
                                                      Resource(edge_cpu_max,
                                                               edge_mem_max,
                                                               edge_cpu_avg,
                                                               edge_mem_avg)),
                                            'edge1',
                                            'default-scheduler'
                                        )), start_time)

                            else:
                                cloud_cpu_max = round((cloud_tasks[temp_real_task_index - half_index].cpu_max + (
                                            1 - random.random())) * cloud_device.cpu, 2)
                                cloud_mem_max = cloud_tasks[temp_real_task_index - half_index].mem_max * cloud_device.mem
                                cloud_cpu_avg = cloud_tasks[temp_real_task_index - half_index].cpu_avg * cloud_device.cpu
                                cloud_mem_avg = cloud_tasks[temp_real_task_index - half_index].mem_avg * cloud_device.mem

                                cpu_container = int(math.ceil(cloud_cpu_max))
                                mem_container = int(math.ceil(cloud_mem_max))

                                tasks_list[node_index] = CPUPod(
                                    Pod(MetaData(job_id, tasks_count, temp_real_task_index, 'cpu', 'cloud'),
                                        TaskSpec(
                                            Container(cpu_container,
                                                mem_container,
                                                int(cloud_tasks[
                                                        temp_real_task_index - half_index].duration) * 1000,
                                                # 毫秒
                                                Resource(
                                                    cloud_cpu_max,
                                                    cloud_mem_max,
                                                    cloud_cpu_avg,
                                                    cloud_mem_avg),
                                            ),
                                            'cloud',
                                            'default-scheduler'
                                        )), start_time)

                    elif scene == 'cloud-edge':
                        edge_title_time = 4000

                        half_index = int(math.floor(len(tasks_dag_new_data) * (1 - task_percentage)))
                        for temp_real_task_index in range(len(tasks_dag_new_data)):
                            dag_task = tasks_dag_new_data[temp_real_task_index].node.split('_')
                            node_index = int(dag_task[0][1:])
                            if len(dag_task) > 0:
                                dag_task = [int(t) for t in dag_task[1:]]
                            start_time = 0
                            start_time2 = 0
                            for t in dag_task:
                                start_time = max(start_time, tasks_list[t].startTime + int(
                                    tasks_list[t].pod.spec.containers[0].args[5]))
                                start_time2 = max(start_time2, tasks_list2[t].startTime + int(
                                    tasks_list2[t].pod.spec.containers[0].args[5]))
                            if temp_real_task_index > half_index:
                                edge_cpu_max = round((edge_tasks[temp_real_task_index - 1 - half_index].cpu_max + (
                                        1 - random.random())) * edge_device.cpu, 2)
                                edge_mem_max = edge_tasks[temp_real_task_index - 1 - half_index].mem_max * edge_device.mem
                                edge_cpu_avg = edge_tasks[temp_real_task_index - 1 - half_index].cpu_avg * edge_device.mem
                                edge_mem_avg = edge_tasks[temp_real_task_index - 1 - half_index].mem_avg * edge_device.mem
                                cpu_container = int(math.ceil(edge_cpu_max))
                                mem_container = int(math.ceil(edge_mem_max))
                                tasks_list[node_index] = CPUPod(
                                    Pod(MetaData(job_id, tasks_count, temp_real_task_index, 'cpu', 'edge1'),
                                        TaskSpec(
                                            Container(cpu_container,
                                                mem_container,
                                                int(edge_tasks[
                                                        temp_real_task_index - 1 - half_index].duration) * edge_title_time,
                                                # 毫秒
                                                Resource(edge_cpu_max,
                                                         edge_mem_max,
                                                         edge_cpu_avg,
                                                         edge_mem_avg),
                                            ), 'edge1',
                                            'default-scheduler')), start_time)

                            else:
                                cloud_cpu_max = round((cloud_tasks[temp_real_task_index].cpu_max + (
                                        1 - random.random())) * cloud_device.cpu, 2)
                                cloud_mem_max = cloud_tasks[
                                                    temp_real_task_index].mem_max * cloud_device.mem
                                cloud_cpu_avg = cloud_tasks[
                                                    temp_real_task_index].cpu_avg * cloud_device.cpu
                                cloud_mem_avg = cloud_tasks[
                                                    temp_real_task_index].mem_avg * cloud_device.mem

                                cpu_container = int(math.ceil(cloud_cpu_max))
                                mem_container = int(math.ceil(cloud_mem_max))
                                tasks_list[node_index] = CPUPod(
                                    Pod(MetaData(job_id, tasks_count, temp_real_task_index, 'cpu', 'cloud'),
                                        TaskSpec(
                                            Container(cpu_container,
                                                      mem_container,
                                                      int(cloud_tasks[temp_real_task_index].duration) * edge_title_time,
                                                      # 毫秒
                                                      Resource(cloud_cpu_max,
                                                               cloud_mem_max,
                                                               cloud_cpu_avg,
                                                               cloud_mem_avg),
                                                      ), 'cloud', 'default-scheduler')), start_time)

                    else:
                        # 分层抽取
                        edge_title_time = 3000
                        edge_tasks_count = 0
                        cloud_tasks_count = 0
                        for temp_real_task_index in range(len(tasks_dag_new_data)):
                            if tasks_dag_new_data[temp_real_task_index].level % MAX_LEVEL_EDGE_CLOUD_EDGE == 2:
                                cloud_tasks_count += 1
                            else:
                                edge_tasks_count += 1

                        edge_tasks.extend(cloud_tasks)
                        edge_tasks.sort(key=reorder2)

                        cloud_tasks = edge_tasks[edge_tasks_count:]
                        random.shuffle(cloud_tasks)
                        edge_tasks = edge_tasks[0: edge_tasks_count]
                        random.shuffle(edge_tasks)
                        i_edge_index = 0
                        i_cloud_index = 0

                        for temp_real_task_index in range(len(tasks_dag_new_data)):
                            dag_task = tasks_dag_new_data[temp_real_task_index].node.split('_')
                            node_index = int(dag_task[0][1:])
                            if len(dag_task) > 0:
                                dag_task = [int(t) for t in dag_task[1:]]
                            start_time = 0
                            start_time2 = 0

                            for t in dag_task:
                                start_time = max(start_time, tasks_list[t].startTime + int(
                                    tasks_list[t].pod.spec.containers[0].args[5]))
                                start_time2 = max(start_time2, tasks_list2[t].startTime + int(
                                    tasks_list2[t].pod.spec.containers[0].args[5]))
                            # 边
                            if tasks_dag_new_data[temp_real_task_index].level % MAX_LEVEL_EDGE_CLOUD_EDGE == 1 or \
                                    tasks_dag_new_data[temp_real_task_index].level % MAX_LEVEL_EDGE_CLOUD_EDGE == 0:
                                edge_cpu_max = round((edge_tasks[i_edge_index].cpu_max + (
                                        1 - random.random())) * edge_device.cpu, 2)
                                edge_mem_max = edge_tasks[i_edge_index].mem_max * edge_device.mem
                                edge_cpu_avg = edge_tasks[i_edge_index].cpu_avg * edge_device.mem
                                edge_mem_avg = edge_tasks[i_edge_index].mem_avg * edge_device.mem
                                cpu_container = int(math.ceil(edge_cpu_max))
                                mem_container = int(math.ceil(edge_mem_max))


                                tasks_list[node_index] = CPUPod(
                                    Pod(MetaData(job_id, tasks_count, temp_real_task_index, 'cpu', 'edge1'),
                                        TaskSpec(
                                            Container(cpu_container,
                                                      mem_container,
                                                      int(edge_tasks[i_edge_index].duration) * edge_title_time,  # 毫秒
                                                      Resource(edge_cpu_max,
                                                               edge_mem_max,
                                                               edge_cpu_avg,
                                                               edge_mem_avg),
                                                      ), 'edge1',
                                            'default-scheduler')), start_time)
                            else:
                                cloud_cpu_max = round((cloud_tasks[i_cloud_index].cpu_max + (
                                        1 - random.random())) * cloud_device.cpu, 2)
                                cloud_mem_max = cloud_tasks[i_cloud_index].mem_max * cloud_device.mem
                                cloud_cpu_avg = cloud_tasks[i_cloud_index].cpu_avg * cloud_device.cpu
                                cloud_mem_avg = cloud_tasks[i_cloud_index].mem_avg * cloud_device.mem

                                cpu_container = int(math.ceil(cloud_cpu_max))
                                mem_container = int(math.ceil(cloud_mem_max))
                                tasks_list[node_index] = CPUPod(
                                    Pod(MetaData(job_id, tasks_count, temp_real_task_index, 'cpu', 'cloud'),
                                        TaskSpec(
                                            Container(cpu_container,
                                                      mem_container,
                                                      int(cloud_tasks[i_cloud_index].duration) * edge_title_time,  # 毫秒
                                                      Resource(cloud_cpu_max,
                                                               cloud_mem_max,
                                                               cloud_cpu_avg,
                                                               cloud_mem_avg),
                                                      ), 'cloud',
                                            'default-scheduler')), start_time)

                                i_cloud_index += 1
                    job_lists.append(tasks_list)
                    job_id += 1
        self.job_lists_answer = job_lists

    def random_jobs(self):
        return self.job_lists_answer
