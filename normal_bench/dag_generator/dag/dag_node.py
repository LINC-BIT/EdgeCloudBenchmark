class DagNode:
    def __init__(self, parent: list, node_name: str, level: int, plan_mem: float, plan_cpu: int):
        self.parent = parent
        self.node_num = node_name
        self.level = level
        self.plan_mem = plan_mem
        self.plan_cpu = plan_cpu
