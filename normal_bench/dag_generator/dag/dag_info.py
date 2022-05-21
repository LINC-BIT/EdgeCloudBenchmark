import pymysql

class DagInfo:
    def __init__(self, tasks: list):
        temp_map = {}
        for task in tasks:
            names = task.task_name.split('_')
            node_name = names[0][1:]
            temp_level = 1
            if len(names) > 1:
                for parent in names[1:]:
                    if temp_map.get(parent) is None:
                        temp_map[node_name] = set(node_name)
                        temp_level = max(temp_level, 2)
                    else:
                        temp_map[node_name].add(node_name)









