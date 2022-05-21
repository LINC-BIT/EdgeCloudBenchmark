import math

from ai_bench.data_generator.const import *


class GPUJob:
    def __init__(self, job_id, spec):
        self.apiVersion = 'batch.volcano.sh/v1alpha1'
        self.kind = 'Job'
        self.metadata = {'labels': {'sub-time': f'{0}'}, 'name': f'job-{job_id}', 'namespace': 'default'}
        self.spec = spec


class Spec:
    def __init__(self, inst_num, tasks):
        self.minAvailable = inst_num
        self.policies = [{'action': 'CompleteJob', 'event': 'TaskCompleted'}]
        self.schedulerName = 'volcano'
        self.tasks = tasks


class GPUTask:
    def __init__(self, inst_num, template):
        self.name = f'test-gpu'
        self.policies = [{'action': 'CompleteJob', 'event': 'TaskCompleted'}]
        self.replicas = inst_num
        self.template = template


class Template:
    def __init__(self, task_spec, no_real_run_time, job_id, job_task_number):
        self.metadata = {'labels': {"app": "linc-workload", "job": f'job-{job_id}', "jobTaskNumber": str(job_task_number), "sim-time": f'{no_real_run_time}'}}
        self.spec = task_spec

"""
volumes:
        - name: share-volume
          persistentVolumeClaim:
            claimName: myclaim
"""

class TaskSpec:
    def __init__(self, containers):
        self.containers = [containers]
        self.restartPolicy = "OnFailure"
        self.volumes = [{"name": "share-volume", "persistentVolumeClaim": {"claimName": "myclaim"}}]


class Container:
    def __init__(self, plan_cpu, plan_mem, plan_gpu, run_time, resource, job_id, task_number):
        self.args = ['--cpu_count', str(plan_cpu), '--gpu_count',
                     str(plan_gpu), '--iter_factor', str(int(run_time)), '--data_dir', '/data', '--job_name', f'gpu-test{job_id}', '--instance_num', str(task_number), '--min_available', str(int(math.ceil(0.6 * task_number)))]
        self.image = 'charles36/workload_test:3.0'
        self.imagePullPolicy = 'IfNotPresent'
        self.name = 'task'
        self.resources = resource
        self.volumeMounts = [{'mountPath': "/data/", "name": "share-volume"}]


    def __str__(self) -> str:
        return str({'args': self.args, 'image': self.image, 'imagePullPolicy': self.imagePullPolicy, 'name': self.name
                , self.resources: self.resources})


class Resource:
    def __init__(self, limits_cpu, limits_memory, requests_cpu, requests_memory, gpu_count):
        self.limits = {'cpu': str(limits_cpu), 'memory': f'{str(limits_memory)}Gi', 'nvidia.com/gpu': gpu_count}
        self.requests = {'cpu': str(requests_cpu), 'memory': f'{str(requests_memory)}Gi'}




# class ContainerArgs:
#     def __init__(self, plan_cpu, plan_mem, plan_gpu, run_time):
#         self.cpu_count = str(plan_cpu)
#         self.plan_mem = str(plan_mem)
#         self.plan_gpu = str(plan_gpu)
#         self.iter_factor = str(int(run_time/20))
#
#     def getMap(self):
#         return ['--cpu_count', self.cpu_count, '--memory-mb',
#                 self.plan_mem, '--gpu_count', self.plan_gpu, '--iter_factor', self.iter_factor]

