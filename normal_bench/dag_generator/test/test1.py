from normal_bench.dag_generator.data import generate

if __name__ == '__main__':
    for i in range(100):
        print(generate.random_job(3))