from normal_bench.generating_component.task_resouce import NormalTaskResource
from normal_bench.generator.const import ALIBABA_CPU_DEVICE, ALIBABA_MEM_DEVICE

class CloudDevice:
    def __init__(self, cpu, mem):
        self.cpu = cpu
        self.mem = mem


class EdgeDevice:
    def __init__(self, cpu, mem):
        self.cpu = cpu
        self.mem = mem


class BenchmarkGenerator:
    # 分别需要传入云边任务比例、设备以及场景
    def __init__(self, task_percentage, device, scene, file_name_job, file_name_tasks, count_jobs):
        self.device = device
        self.task_percentage = task_percentage
        self.scene = scene
        self.file_name_tasks = file_name_tasks
        self.file_name_job = file_name_job
        self.count_jobs = count_jobs
        cloud_device = CloudDevice(int(device['cloud']['cpu']) / ALIBABA_CPU_DEVICE, int(device['cloud']['mem']) / ALIBABA_MEM_DEVICE)
        edge_device = EdgeDevice(int(device['edge']['cpu']) / ALIBABA_CPU_DEVICE, int(device['edge']['mem']) / ALIBABA_MEM_DEVICE)

        self.jobs = NormalTaskResource(file_name_job, file_name_tasks, count_jobs,
                                       scene, task_percentage, edge_device, cloud_device)

    def random_job(self):
        return self.jobs
