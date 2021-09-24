from PIL import Image
import numpy as np
from typing import Any, Callable, List, Tuple, Union
import math


# весовая функция для n-ребра, рассчитанная по значению двух соседних пикселей
# цветные картинки с 3 каналами или одноцветные с 1 каналом
# чем ближе цвета пикселей друг к другу, тем больше вес 
def n_weight(pixel1, pixel2, sigma: float, is_bw: bool):
    if (is_bw == True):
        delta = abs(pixel1 - pixel2)**2       
    else:
        delta = abs(pixel1[0] - pixel2[0])**2 + abs(pixel1[1] - pixel2[1])**2 + abs(pixel1[2] - pixel2[2])**2
    return math.exp(-delta /(2 * sigma**2))



# создает массив пикселей из img из массивов координат по x, y
def create_pixel_list(img, obj_pixels_x: List[int], obj_pixels_y: List[int]):
    if len(obj_pixels_x) !=  len(obj_pixels_y):
        print("List sizes are not consistent")
        return None
    pixel_list = []
    for i in range(len(obj_pixels_x)):
        pixel_list.append(img[obj_pixels_x[i], obj_pixels_y[i]])
    return pixel_list


# возвращает взвешенный граф, построенный по изображению в файле filepath
# obj_pixel - пискель, точно принадлежащий объекту
# bck_pixel - пискель, точно принадлежащий фону
# граф в виде массиве [n, 3], где n - общее число ребер
def get_graph(filepath: str, obj_pixels_x: List[int], obj_pixels_y: List[int], bck_pixels_x: List[int], bck_pixels_y: List[int], _lambda: float = 100.0, sigma = 1.0, is_bw = True):
    # пытаемся открыть файл с изображением
    try:  
        img_file = Image.open(filepath)  
    except FileNotFoundError:  
        print("file not found")
        return None

    img = img_file.load()        # загружаем изображение в память
    height = img_file.height;    # высота картинки в пикселях
    width = img_file.width;      # ширина картинки в пикселях

    img_file.close()

    obj_pixels = create_pixel_list(img, obj_pixels_x, obj_pixels_y)  # создаем массив пикселей объекта
    bck_pixels = create_pixel_list(img, bck_pixels_x, bck_pixels_y)  # создаем массив пикселей фона

    internal_ribs = (width - 2) * (height - 2) * 4                # количество ребер из внутренних пикселей 
    boundary_ribs = (width - 2) * 6 + (height - 2) * 6 + 4 * 2    # количество ребер из граничных пикселей  
    s_ribs = width * height                                       # количество ребер из источника в пиксели
    t_ribs = width * height                                       # количество ребер из пискелей в сток 
    total_ribs = internal_ribs + boundary_ribs + s_ribs + t_ribs  # суммарное количество ребер в графе

    # rib_list = np.zeros((total_ribs, 3), np.int32) #выделяем память под список ребер
    rib_list = [[0, 0, 0] for i in range(total_ribs)]

    # итерируемся по внутренним пикселям, где каждый пискель имеет 4 соседа
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            px = i * width + j + 1                                        # индекс центрального пикселя
            cnt = s_ribs + t_ribs + ((i - 1) * (width - 2) + j - 1) * 4   # счетчик для индексации ребер

            # ребра из центрального пикселя в левый
            rib_list[cnt][0] = px
            rib_list[cnt][1] = px - 1    
            rib_list[cnt][2] = n_weight(img[j, i], img[j - 1, i], sigma, is_bw) 
            cnt += 1

            # ребра из центрального пикселя в правый
            rib_list[cnt][0] = px
            rib_list[cnt][1] = px + 1
            rib_list[cnt][2] = n_weight(img[j, i], img[j + 1, i], sigma, is_bw) 
            cnt += 1

            # ребра из центрального пикселя в верхний
            rib_list[cnt][0] = px
            rib_list[cnt][1] = px - width
            rib_list[cnt][2] = n_weight(img[j, i], img[j, i], sigma, is_bw) 
            cnt += 1

            # ребра из центрального пикселя в нижний
            rib_list[cnt][0] = px
            rib_list[cnt][1] = px + width
            rib_list[cnt][2] = n_weight(img[j, i], img[j, i], sigma, is_bw)

    # итерируемся по горизонтальным границам, где каждый пискель имеет 3 соседа
    for i in range(1, width - 1): 
        cnt = internal_ribs + s_ribs + t_ribs + (i - 1) * 6  # счетчик для индексации ребер 

        # ребра из центрального пикселя верхней границы в левый
        rib_list[cnt][0] = i + 1 
        rib_list[cnt][1] = i     
        rib_list[cnt][2] = n_weight(img[i, 0], img[i - 1, 0], sigma, is_bw) 
        cnt += 1
        # ребра из центрального пикселя верхней границы в правый
        rib_list[cnt][0] = i + 1
        rib_list[cnt][1] = i + 2
        rib_list[cnt][2] = n_weight(img[i, 0], img[i + 1, 0], sigma, is_bw) 
        cnt += 1
        # ребра из центрального пикселя верхней границы в нижний
        rib_list[cnt][0] = i + 1
        rib_list[cnt][1] = i + width + 1
        rib_list[cnt][2] = n_weight(img[i, 0], img[i, 1], sigma, is_bw) 
        cnt += 1

        index = (height - 1) * width + i + 1
        # ребра из центрального пикселя нижней границы в левый
        rib_list[cnt][0] = index
        rib_list[cnt][1] = index - 1 
        rib_list[cnt][2] = n_weight(img[i, height - 1], img[i - 1, height - 1], sigma, is_bw) 
        cnt += 1
        # ребра из центрального пикселя нижней границыв правый
        rib_list[cnt][0] = index 
        rib_list[cnt][1] = index + 1
        rib_list[cnt][2] = n_weight(img[i, height - 1], img[i + 1, height - 1], sigma, is_bw) 
        cnt += 1
        # ребра из центрального пикселя нижней границы в верхний
        rib_list[cnt][0] = index
        rib_list[cnt][1] = index - width
        rib_list[cnt][2] = n_weight(img[i, height - 1], img[i, height - 2], sigma, is_bw) 
    
    # итерируемся по вертикальным границам, где каждый пискель имеет 3 соседа
    for i in range(1, height - 1): 
        cnt = internal_ribs + s_ribs + t_ribs + (width - 2) * 6 + (i - 1) * 6  # счетчик для индексации ребер 

        # ребра из центрального пикселя левой границы в верхний
        rib_list[cnt][0] = i * width + 1
        rib_list[cnt][1] = (i - 1) * width + 1   
        rib_list[cnt][2] = n_weight(img[0, i], img[0, i - 1], sigma, is_bw) 
        cnt += 1
        # ребра из центрального пикселя левой границы в нижний
        rib_list[cnt][0] = i * width + 1
        rib_list[cnt][1] = (i + 1) * width + 1  
        rib_list[cnt][2] = n_weight(img[0, i], img[0, i + 1], sigma, is_bw) 
        cnt += 1
        # ребра из центрального пикселя левой границы в правый
        rib_list[cnt][0] = i * width + 1
        rib_list[cnt][1] = i + width + 2
        rib_list[cnt][2] = n_weight(img[0, i], img[1, i], sigma, is_bw) 
        cnt += 1

        # ребра из центрального пикселя правой границы в верхний
        rib_list[cnt][0] = (i + 1) * width
        rib_list[cnt][1] = i * width
        rib_list[cnt][2] = n_weight(img[width - 1, i], img[width - 1, i - 1], sigma, is_bw) 
        cnt += 1
        # ребра из центрального пикселя правой границы в нижний
        rib_list[cnt][0] = (i + 1) * width
        rib_list[cnt][1] = (i + 2) * width
        rib_list[cnt][2] = n_weight(img[width - 1, i], img[width - 1, i + 1], sigma, is_bw) 
        cnt += 1
        # ребра из центрального пикселя нправой границы в левый
        rib_list[cnt][0] = (i + 1) * width
        rib_list[cnt][1] = (i + 1) * width - 1
        rib_list[cnt][2] = n_weight(img[width - 1, i], img[width - 2, i], sigma, is_bw)        

    #верхний левый угловой пискель
    rib_list[total_ribs - 8][0] = 1
    rib_list[total_ribs - 8][1] = 2
    rib_list[total_ribs - 8][2] = n_weight(img[0, 0], img[1, 0], sigma, is_bw)   

    rib_list[total_ribs - 7][0] = 1
    rib_list[total_ribs - 7][1] = 1 + width
    rib_list[total_ribs - 7][2] = n_weight(img[0, 0], img[0, 1], sigma, is_bw)   

    #верхний правый угловой пискель
    rib_list[total_ribs - 6][0] = width
    rib_list[total_ribs - 6][1] = width - 1
    rib_list[total_ribs - 6][2] = n_weight(img[width - 1, 0], img[width - 2, 0], sigma, is_bw)   

    rib_list[total_ribs - 5][0] = width
    rib_list[total_ribs - 5][1] = 2 * width 
    rib_list[total_ribs - 5][2] = n_weight(img[width - 1, 0], img[width - 1, 1], sigma, is_bw)    

    #нижний левый угловой пискель
    rib_list[total_ribs - 4][0] = (height - 1) * width + 1
    rib_list[total_ribs - 4][1] = (height - 2) * width + 1
    rib_list[total_ribs - 4][2] = n_weight(img[0, height - 1], img[0, height - 2], sigma, is_bw)   

    rib_list[total_ribs - 3][0] = (height - 1) * width + 1
    rib_list[total_ribs - 3][1] = (height - 1) * width + 2
    rib_list[total_ribs - 3][2] = n_weight(img[0, height - 1], img[1, height - 1], sigma, is_bw)   

    #нижний правый угловой пискель
    rib_list[total_ribs - 2][0] = height * width
    rib_list[total_ribs - 2][1] = (height - 1) * width
    rib_list[total_ribs - 2][2] = n_weight(img[width - 1, height - 1], img[width - 1, height - 2], sigma, is_bw)   

    rib_list[total_ribs - 1][0] = height * width
    rib_list[total_ribs - 1][1] = height * width - 1
    rib_list[total_ribs - 1][2] = n_weight(img[width - 1, height - 1], img[width - 2, height - 1], sigma, is_bw)  


    out_flow: List[int] = [0] * (height * width + 1) # массив суммарных пропускных способностей вершин
    for arc in rib_list:
        out_flow[arc[0]] += arc[2]
    K: float = max(out_flow) + 1.0 # максимальная пропускная способность вершины графа + 1
    out_flow.clear()

    R_obj = get_histogram_distribution(obj_pixels, K, _lambda)
    R_bck = get_histogram_distribution(bck_pixels, K, _lambda)

    def does_index_belong_to(index_x: int, index_y: int, index_list_x: List[int], index_list_y: List[int]):
        does_it: bool = False
        for i in range(len(index_list_x)):
            if index_x == index_list_x[i] and index_y == index_list_y[i]:
                does_it = True
                break
        return does_it

    # итерируемся по всем пикселям для задания t-связей
    for i in range(height):
        for j in range(width):
            index = i * width + j
            rib_list[index][0] = 0          # источник является вершиной 0
            rib_list[index][1] = index + 1  # пиксели как вершины

            rib_list[index + width * height][0] = index + 1           # пиксели как вершины
            rib_list[index + width * height][1] = height * width + 1  # сток является последней вершиной

            if does_index_belong_to(j, i, obj_pixels_x, obj_pixels_y):
                rib_list[index][2] = K
                rib_list[index + width * height][2] = 0
            elif does_index_belong_to(j, i, bck_pixels_x, bck_pixels_y):
                rib_list[index][2] = 0
                rib_list[index + width * height][2] = K
            else:
                rib_list[index][2] = R_bck(img[j, i])
                rib_list[index + width * height][2] = R_obj(img[j, i])

    return rib_list, height, width, K


def add_new_seeds(
    rib_list: np.ndarray, width: int, height: int, K: int,
    obj_pixels_x: List[int], obj_pixels_y: List[int],
    bck_pixels_x: List[int], bck_pixels_y: List[int]
) -> None:
    """
    Updates the weights according to new pixel seeds.

    Parameters
    ----------
    rib_list: np.ndarray
        numpy array [v,3] - [[node1, node2, capacity]]
    width: int
        image width
    height: int
        image height
    K: int
        number K used in the algorithm
    obj_pixels_x: List[int]
        x coordinate of new object pixel
    obj_pixels_y: List[int]
        y coordinate of new object pixel
    bck_pixels_x: List[int]
        x coordinate of new background pixel
    bck_pixels_y: List[int]
        y coordinate of new background pixel

    Updates rib_list and returns nothing.
    """

    for i in range(len(obj_pixels_x)):
        index = obj_pixels_y[i] * width + obj_pixels_x[i]

        rib_list[index][2] = max(rib_list[index][2], rib_list[index + width * height][2] + K)

    for i in range(len(bck_pixels_x)):
        index = bck_pixels_y[i] * width + bck_pixels_x[i]

        rib_list[index + width * height][2] = max(rib_list[index + width * height][2], rib_list[index][2] + K)


def get_bwimage(w_pixels: List[int], width: int, height: int) -> Image.Image:
    """
    Construct a black and white image from the given list of white pixels.

    Parameters
    ----------
    cut: List[Node]
        list of white pixels indicies starting from one
    width: int
        width of result image
    height: int
        height of result image

    Returns a [width x height] image with white pixels at the given positions and black pixels elsewhere.
    """

    image = Image.new("1", (width, height))
    for i in w_pixels:
        image.putpixel(((i - 1) % width, (i - 1) // width), 1)
    return image


def get_histogram_distribution(pixels: List[int], K: int, lambda_mul: float = 1.0) -> Callable[[int], int]:
    """
    Construct a (lambda-scaled) log probability distribution of pixel intensity.

    Parameters
    ----------
    pixels: List[Tuple[int, int]]
        list of pixel intensities used in the histogram construction
    lambda_mul: float = 1.0
        scaling multiplier

    Returns a (lambda-scaled) log probability function of pixel intensity.
    """

    pixels_number: int = len(pixels)
    # groups_number = math.floor(math.log(pixels_number)) + 1
    groups_number: int = 51
    groups: List[int] = [0 for i in range(groups_number)]
    mult: float = groups_number / 256

    for intensity in pixels:
        groups[math.floor(intensity * mult)] += 1

    for i in range(groups_number):
        if groups[i] != 0:
            groups[i] = - lambda_mul * math.log(groups[i] / pixels_number)
        else:
            groups[i] = K

    def distribution(intensity: int) -> int:
        """Closure function of probabilistic distribution."""

        nonlocal groups
        nonlocal mult

        scaled_intensity: int = math.floor(intensity * mult)
        return groups[scaled_intensity]

    return distribution
