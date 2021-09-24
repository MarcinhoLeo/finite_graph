from PIL import Image
import numpy as np


def quality_metrics(img_file1: Image.Image, img_file2: Image.Image) -> None:
    height1 = img_file1.height;    # высота картинки в пикселях
    width1 = img_file1.width;      # ширина картинки в пикселях
    img1 = img_file1.load()       # загружаем изображение в память

    height2 = img_file2.height;    # высота картинки в пикселях
    width2 = img_file2.width;      # ширина картинки в пикселях
    img2 = img_file2.load() # загружаем изображение в память

    if (height1 != height2 or width1 != width2):
        print("error: images have different size")
        return

    obj_correct_pixels = 0;  # количество совпадающих пикселей в пересечении пикселей объектов 
    bck_correct_pixels = 0   # количество совпадающих пикселей в пересечении пикселей фонов
    obj_union_count = 0;     # количество закрашенных пикселей в объединении пикселей объектов

    for i in range(width1):
        for j in range(height1):
            pixel1 = img1[i, j]
            pixel2 = img2[i, j]

            if (isinstance(pixel1, int) == True):
                intensity1 = pixel1
            else:
                intensity1 = pixel1[0] + pixel1[1] + pixel1[2]
            if (isinstance(pixel2, int) == True):
                intensity2 = pixel2
            else:
                intensity2 = pixel2[0] + pixel2[1] + pixel2[2]

            if (intensity1 > 0 and intensity2 > 0):
                obj_correct_pixels += 1
            if (intensity1 > 0 or  intensity2 > 0):
                obj_union_count += 1
            if (intensity1 == 0 and intensity2 == 0):
                bck_correct_pixels += 1

    print('ratio of correct object pixels to all', obj_correct_pixels / (width1 * height1))
    print('ratio of correct background pixels to all', bck_correct_pixels / (width1 * height1))
    print('we got sum: ', (obj_correct_pixels + bck_correct_pixels) / (width1 * height1))
    print('cardinality ratio of intersection to union is' , obj_correct_pixels / obj_union_count)
