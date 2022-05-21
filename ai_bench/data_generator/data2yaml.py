# -*- coding: UTF-8 -*-

import pymysql
import matplotlib.pyplot as plt
import math
import yaml

def generate_yaml_doc(yaml_file,name="job" , job_id=1, plan_cpu=1, plan_mem=512, plan_gpu=1, run_time=10 ,inst_num=1):

    pod0={}
    pod0["apiVersion"]="batch.volcano.sh/v1alpha1"
    pod0["kind"] = "Job"
    metadata0={}
    labels0={}
    metadata0["name"]=name+"-"+str(job_id)
    metadata0["namespace"] = "default"
    pod0["metadata"]=metadata0
    spec0={}
    spec0["minAvailable"]=inst_num
    spec0["schedulerName"]="volcano"
    policies0=[]
    policies_dic={}#带_dic表示这个字典要append到一个list中
    policies_dic["event"]="TaskCompleted"
    policies_dic["action"] = "CompleteJob"
    policies0.append(policies_dic)
    spec0["policies"]=policies0
    tasks0=[]
    task_dic={}
    task_dic["replicas"]=inst_num
    task_dic["name"]="test-gpu"
    policies0=[]
    policies_dic={}
    policies_dic["event"]="TaskCompleted"
    policies_dic["action"] = "CompleteJob"
    policies0.append(policies_dic)
    task_dic["policies"]=policies0
    template0={}
    containers0=[]
    container_dic = {}
    args0=[] #args为container运行程序附带参数
    args0.append("--cpu_count")
    args0.append(str(plan_cpu))
    args0.append("--gpu_count")
    args0.append(str(plan_gpu))
    args0.append("--iter_factor") #1对应2秒
    args0.append(str(int(run_time/20)))
    container_dic["args"]=args0
    container_dic["image"]="charles36/workload_test:latest"
    container_dic["name"]="task" #task_name
    resources0={}
    limits0={}
    limits0["cpu"]=str(2*plan_cpu)
    limits0["memory"]=str(2*plan_mem)+"Mi"
    limits0["nvidia.com/gpu"]=plan_gpu
    resources0["limits"]=limits0
    requests0={}
    requests0["cpu"]=str(plan_cpu) #plan_cpu
    requests0["memory"]=str(plan_mem)+"Mi" #plan_mem
    resources0["requests"]=requests0
    container_dic["resources"]=resources0
    container_dic["imagePullPolicy"]="IfNotPresent"
    containers0.append(container_dic)
    spec1={}
    spec1["containers"]=containers0
    spec1["restartPolicy"] = "OnFailure"
    template0["spec"]=spec1
    task_dic["template"]=template0
    tasks0.append(task_dic)
    spec0["tasks"]=tasks0
    pod0["spec"]=spec0


    file=open(yaml_file,'w',encoding='utf-8')
    yaml.dump(pod0,file)
    file.close()

host = '10.1.114.52'
port = 9030
db = 'ai_bench'
user = 'root'
password = 'dhj19991101'


conn = pymysql.connect(host=host, port=port, db=db, user=user, password=password)

#按每次read值进行分类
# 使用 cursor() 方法创建一个 dict 格式的游标对象 cursor
cursor = conn.cursor(pymysql.cursors.DictCursor)

#查可用项数(不为null)
cursor.execute("select pai_instance_table.task_name task_name, pai_instance_table.start_time as start_time, "
               "pai_instance_table.end_time as end_time, "
               "plan_cpu, plan_gpu, plan_mem, inst_num "
               "from pai_task_table inner join pai_instance_table "
               "on pai_instance_table.job_name=pai_task_table.job_name "
               "and pai_instance_table.task_name=pai_task_table.task_name "
               "where pai_instance_table.start_time is not null and plan_cpu is not null "
               "and plan_gpu is not null and plan_mem is not null "
               "and pai_instance_table.end_time is not null and plan_cpu<2000 "
               "and inst_num<10 and pai_instance_table.end_time-pai_instance_table.start_time<2500 "
               "order by rand() "
               "limit 40")
data = cursor.fetchall()
for i,d in enumerate(data):
    # inst_num = math.floor(d["inst_num"])
    # run_time = math.floor(d["end_time"]-d["start_time"])
    # plan_cpu = math.floor( d["plan_cpu"]/100 ) #math.floor向下取整
    # plan_gpu = math.floor( d["plan_gpu"]/100 )
    # plan_mem = math.floor( d["plan_mem"]*1024 ) #乘后为MB
    inst_num = round(d["inst_num"])
    run_time = round(d["end_time"] - d["start_time"])
    plan_cpu = round(d["plan_cpu"] / 100)  # math.floor向下取整
    plan_gpu = round(d["plan_gpu"] / 100)
    plan_mem = round(d["plan_mem"] * 1024)  # 乘后为MB
    print(d["task_name"],inst_num,plan_cpu,plan_gpu, plan_mem, run_time)
    generate_yaml_doc("workloads7/test-"+str(i)+".yaml",name="gpu-test", job_id=i,run_time=run_time,
                      plan_cpu=plan_cpu, plan_mem=plan_mem, plan_gpu=plan_gpu, inst_num=inst_num)



# 关闭数据库连接
cursor.close()
conn.close()