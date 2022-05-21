from random import shuffle as sl
from random import randint as rd


def write_to_file(f, num, fg):
    f.write(str(num))
    if fg:
        f.write('\n')
    else:
        f.write(' ')


class RandomGraph:
    def __init__(self, file_name, edge_count):
        self.file_name = file_name
        self.edge_count = edge_count

    #
    @staticmethod
    def generate_weight():
        num = rd(1, 1000)
        return num


    def data_make(self):
        f = open(f'{self.file_name}.in', 'w')
        node = list(range(1, self.edge_count + 1))
        sl(node)
        sl(node)
        print(node)
        m = rd(1, min(self.edge_count * (self.edge_count - 1) / 2, 5000))
        write_to_file(f, self.edge_count, 1)
        temp_set = set()
        for i in range(0, m):
            p1 = rd(1, self.edge_count - 1)
            p2 = rd(p1 + 1, self.edge_count)
            x = node[p1 - 1]
            y = node[p2 - 1]
            while f'{x}-{y}' in temp_set:
                p1 = rd(1, self.edge_count - 1)
                p2 = rd(p1 + 1, self.edge_count)
                x = node[p1 - 1]
                y = node[p2 - 1]
            temp_set.add(f'{x}-{y}')
            write_to_file(f, y, 0)
            write_to_file(f, x, 1)
        print(self.edge_count, ' node', m, ' edges')
        f.close()
        print('Done')
