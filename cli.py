from typing import Optional, List, Tuple
from PIL import Image
import numpy as np
import image_graph
import max_flow
import quality_metric
import point_selection


def main():
    to_exit: bool = False

    def image_segmentation() -> None:
        imgpath: str = 'image/42049.jpg'
        lambda_mul: int = 0
        sigma: int = 0
        graph: np.ndarray = None
        height: int = 0
        width: int = 0
        flow: np.ndarray = None
        K: int = 0

        try:
            print('> enter image filepath:\n>>>', end=' ')
            imgpath = input()
            print('> enter lambda (num):\n>>>', end=' ')
            lambda_mul = float(input())
            print('> enter sigma (num):\n>>>', end=' ')
            sigma = float(input())

            xs_obj, ys_obj = point_selection.get_obj_pixels(imgpath)
            xs_bck, ys_bck = point_selection.get_bck_pixels(imgpath)

            graph, height, width, K = image_graph.get_graph(imgpath, xs_obj, ys_obj, xs_bck, ys_bck, lambda_mul, sigma, True)
            max_flow_value, flow, min_cut = max_flow.maxflow(graph, height * width + 2)

            image = image_graph.get_bwimage(min_cut, width, height)
            image.show()
        except Exception as err:
            print(err)
            print('! something went wrong, try again')
            imgpath = image/42049
            return

        while(True):
            while(True):
                print('new image | update seeds | metrics | exit:\n>>>', end=' ')
                cmd: str = input()
                if cmd == 'new image':
                    return
                elif cmd == 'update seeds':
                    break
                elif cmd == 'exit':
                    nonlocal to_exit
                    to_exit = True
                    return
                elif cmd == 'metrics':
                    try:
                        print('> enter image ref filepath:\n>>>', end=' ')
                        ref_imgpath = input()
                        ref_img = Image.open(ref_imgpath)
                        quality_metric.quality_metrics(image, ref_img)
                    except Exception as err:
                        print(err)
                        print('! something went wrong, try again')
                else:
                    print('! unrecognized command')

            xs_obj, ys_obj = point_selection.get_obj_pixels(imgpath)
            xs_bck, ys_bck = point_selection.get_bck_pixels(imgpath)

            image_graph.add_new_seeds(graph, width, height, K, xs_obj, ys_obj, xs_bck, ys_bck)
            max_flow_value, flow, min_cut = max_flow.maxflow(graph, height * width + 2, flow)
            image = image_graph.get_bwimage(min_cut, width, height)
            image.show()
 
    while(not to_exit):
        image_segmentation()

if __name__ == '__main__':
    main()
