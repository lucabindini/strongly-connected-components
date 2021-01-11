import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer

sys.setrecursionlimit(100000)

time = 0


class Node:
    def __init__(self, value):
        self.id = value
        self.d = 0
        self.f = 0
        self.pi = None
        self.color = 'white'


def dfs(matrix, nodes):
    global time
    time = 0
    i = 0
    for u in range(len(nodes)):
        if nodes[u].color == 'white':
            dfs_visit(matrix, nodes, u, 'false')
            i += 1
    return i


def dfs_transpose(matrix, nodes, nodes_ordered):
    global time
    time = 0
    i = 0
    for u in range(len(nodes_ordered)):
        if nodes[nodes_ordered[u].id].color == 'white':
            print("{}^a componente fortemente connessa:".format(i + 1))
            print(dfs_visit(matrix, nodes, nodes_ordered[u].id, 'true'))
            i += 1
    return i


def dfs_visit(matrix, nodes, u, is_scc):
    global time
    time += 1
    nodes[u].d = time
    nodes[u].color = 'gray'
    component = [u]
    for v in range(len(nodes)):
        if matrix[u][v] != 0 and nodes[v].color == 'white':
            nodes[v].pi = nodes[u]
            component_return = dfs_visit(matrix, nodes, v, is_scc)
            if is_scc == 'true':
                for i in range(len(component_return)):
                    component.append(component_return[i])
    nodes[u].color = 'black'
    time += 1
    nodes[u].f = time
    if is_scc == 'true':
        component.sort()
        return component


def scc(matrix, nodes):
    dfs(matrix, nodes)
    matrix_t = matrix.transpose()
    nodes_sorted = nodes[:]
    nodes_sorted.sort(key=lambda x: x.f)
    nodes_sorted.reverse()
    print_nodes(nodes_sorted)
    for u in range(len(nodes)):
        nodes[u].color = 'white'
        nodes[u].pi = None
    return dfs_transpose(matrix_t, nodes, nodes_sorted)


def dfs_test(matrix, nodes):
    start = timer()
    count = dfs(matrix, nodes)
    end = timer()
    time_dfs = end - start
    print("Tempo di esecuzione DFS:", time_dfs)
    return count


def scc_test(matrix, nodes):
    start = timer()
    count = scc(matrix, nodes)
    end = timer()
    time_scc = end - start
    print("Tempo di esecuzione SCC:", time_scc)
    return count


def adiacent_matrix_creation(size, prob):
    matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i != j and random.randint(1, 100) <= prob:
                matrix[i][j] = 1
    print(matrix)
    return matrix


def print_nodes(nodes):
    for i in range(len(nodes)):
        print("Nodo", nodes[i].id, "u.d", nodes[i].d, "u.f", nodes[i].f)


def avg(array):
    avg = 0
    for i in range(len(array)):
        avg += array[i]
    return \
        float(avg) / float(len(array))


def main():
    dim = 100
    dfs_num = []
    scc_num = []
    probability = []
    for prob in range(0, 51):
        probability.append(prob)
        dfs_single = []
        scc_single = []
        print("Dim:", dim, "Prob:", prob)
        for n in range(0, 10):
            nodes = []
            for k in range(dim):
                nodes.append(Node(k))
            matrix = adiacent_matrix_creation(dim, prob)
            dfs_single.append(dfs_test(matrix, nodes))
            scc_single.append(scc_test(matrix, nodes))
        dfs_num.append(avg(dfs_single))
        scc_num.append(avg(scc_single))
    plt.plot(probability, scc_num)
    plt.xlabel('Probabilita')
    plt.ylabel('Numero SCC')
    plt.legend(['SCC'])
    plt.savefig('SCC')
    plt.clf()
    plt.plot(probability, dfs_num)
    plt.plot(probability, scc_num)
    plt.legend(['Primo Attraversamento DFS', 'Secondo Attraversamento DFS (SCC)'])
    plt.xlabel('Probabilita')
    plt.ylabel('Numero radici')
    plt.savefig('DFS')
    plt.clf()


if __name__ == '__main__':
    main()
