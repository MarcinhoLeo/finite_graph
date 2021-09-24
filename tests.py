import time
import numpy as np
import max_flow


# считать список ребер графа, прочитанных из файла filepath
# вовзращает список ребер и количество вершин
# список ребер представлен как numpy массив [m, 3]
def read_graph(filepath):
    try:
        file = open(filepath, "r")
    except FileNotFoundError:
        print("file not found")
        return None

    first_line = file.readline().split(' ') # считываем первую строку с количеством вершин и ребер
    n = int(first_line[0])                  # количество вершин
    m = int(first_line[1])                  # количество ребер

    rib_list = np.zeros((m, 3), np.int32)   # выделяем память под список ребер

    # читаем строки из файла и извлекаем из них значения ребер
    for i in range(m):
        line = file.readline().split(' ')    # читаем и разбиваем текущую строку на токены
        rib_list[i] = [int(x) for x in line] # заполняем текущее ребро

        # в файле вершин нумерация с 1
        rib_list[i][0] -= 1;
        rib_list[i][1] -= 1;

    file.close()
    return rib_list, n


# тестирует реализацию алгоритма максимального потока 2 способами
# читает граф из файла filepath и замеряет время работы на нем
# выводит значение макс потока
def file_test(filepath):
    graph, n = read_graph(filepath)
    print("file test: ", filepath)

    start = time.time()
    flow_val, flow = max_flow.maxflow(graph, n, find_mincut=False)
    end = time.time()
    print("time elapsed in seconds", "%.3f" %(end - start))  # время выполнения с точностью 3 знака после запятой
    print("max flow:", flow_val)


if __name__ == "__main__":
    # tests 1-6
    file_test("MaxFlow-tests/test_1.txt")
    file_test("MaxFlow-tests/test_2.txt")
    file_test("MaxFlow-tests/test_3.txt")
    file_test("MaxFlow-tests/test_4.txt")
    file_test("MaxFlow-tests/test_5.txt")
    file_test("MaxFlow-tests/test_6.txt")

    # tests d1-d5
    file_test("MaxFlow-tests/test_d1.txt")
    file_test("MaxFlow-tests/test_d2.txt")
    file_test("MaxFlow-tests/test_d3.txt")
    file_test("MaxFlow-tests/test_d4.txt")
    file_test("MaxFlow-tests/test_d5.txt")

    # tests rd1-rd7
    file_test("MaxFlow-tests/test_rd01.txt")
    file_test("MaxFlow-tests/test_rd02.txt")
    file_test("MaxFlow-tests/test_rd03.txt")
    file_test("MaxFlow-tests/test_rd04.txt")
    file_test("MaxFlow-tests/test_rd05.txt")
    file_test("MaxFlow-tests/test_rd06.txt")
    file_test("MaxFlow-tests/test_rd07.txt")

    # tests rl1-rl10
    file_test("MaxFlow-tests/test_rl01.txt")
    file_test("MaxFlow-tests/test_rl02.txt")
    file_test("MaxFlow-tests/test_rl03.txt")
    file_test("MaxFlow-tests/test_rl04.txt")
    file_test("MaxFlow-tests/test_rl05.txt")
    file_test("MaxFlow-tests/test_rl06.txt")
    file_test("MaxFlow-tests/test_rl07.txt")
    file_test("MaxFlow-tests/test_rl08.txt")
    file_test("MaxFlow-tests/test_rl09.txt")
    file_test("MaxFlow-tests/test_rl10.txt")
