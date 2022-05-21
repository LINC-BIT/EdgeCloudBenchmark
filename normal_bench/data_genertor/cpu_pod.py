from normal_bench.data_genertor.const import *


class CPUPod:
    def __init__(self, pod, start_time):
        self.pod = pod
        self.startTime = start_time


class Pod:
    def __init__(self, meta_data, task_spec):
        self.apiVersion = API_VERSION
        self.kind = 'Pod'
        self.metadata = meta_data
        self.spec = task_spec


class MetaData:
    def __init__(self, job_id, job_task_number, task_id, task_type, multi_type):
        self.labels = {'app': APP, 'job': f'job-{job_id}',
                       'jobTaskNumber': f'n{job_task_number}',
                       'taskType': task_type}
        self.name = f'job-{job_id}-{multi_type}-{task_id}'
        self.namespace = NAME_SPACE


class TaskSpec:
    def __init__(self, containers, node_type, scheduler_name):
        self.containers = [containers]
        self.restartPolicy = RESTART_POLICY
        self.imagePullPolicy = IMAGE_PULL_POLICY
        self.nodeSelector = {'linc/nodeType': node_type}
        self.schedulerName = scheduler_name


class Container:
    def __init__(self, plan_cpu, plan_mem, run_time, resource):
        self.args = ['--cpu-count', str(plan_cpu), '--memory-mb', str(plan_mem), '--iter-factor', str(int(run_time/20))]
        self.image = IMAGE
        self.name = 'task'
        self.resources = resource



class Resource:
    def __init__(self, limits_cpu, limits_memory, requests_cpu, requests_memory):
        self.limits = {'cpu': str(limits_cpu), 'memory': f'{str(limits_memory)}Mi'}
        self.requests = {'cpu': str(requests_cpu), 'memory': f'{str(requests_memory)}Mi'}



