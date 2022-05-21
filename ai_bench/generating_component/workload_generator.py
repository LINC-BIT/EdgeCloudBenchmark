from ai_bench.generating_component.task_resource import TaskResource
from ai_bench.generator.const import ALIBABA_CPU_DEVICE, ALIBABA_MEM_DEVICE, ALIBABA_GPU_DEVICE


class CloudDevice:
    def __init__(self, cpu, mem, gpu):
        self.cpu = cpu
        self.mem = mem
        self.gpu = gpu


class AIBenchmarkGenerator:
    def __init__(self, task_percentage, device, scene, file_name_job, file_name_tasks, count_jobs):
        self.device = device
        self.task_percentage = task_percentage
        self.scene = scene
        self.file_name_tasks = file_name_tasks
        self.file_name_job = file_name_job
        self.count_jobs = count_jobs
        cloud_device = CloudDevice(int(device['cloud']['cpu']) / ALIBABA_CPU_DEVICE, int(device['cloud']['mem']) / ALIBABA_MEM_DEVICE)

        self.jobs = TaskResource(file_name_job, file_name_tasks, count_jobs, cloud_device)

    def random_job(self):
        return self.jobs

