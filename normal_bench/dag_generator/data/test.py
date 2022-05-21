from normal_bench.dag_generator.data.generate import random_job

a = random_job(10)
print(a)


def findCriPath(new_tasks):
    temp_tasks = new_tasks['tasks']
    # 对整个tasks进行标号
    a = [1 for i in range(len(temp_tasks))]
    index = 0
    max_value = 1
    for temp_task in temp_tasks:

        tt = temp_task.split('_')
        if len(tt) > 1:
            for temp_task_1 in tt[1:]:
                a[index] = max(a[int(temp_task_1)] + 1, a[index])
                max_value = max(max_value, a[index])
        index += 1

    return max_value

print(findCriPath(a))

