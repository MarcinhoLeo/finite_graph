from numpy.lib.type_check import imag
from PIL import Image
import image_graph
import max_flow
import quality_metric
import point_selection



# получить сегментацию изображения source_img_path
# результат сохраняется в result_path
# is_bw - является ли изображение черно-белым
def get_segmentation(source_img_path: str, _lambda: int, sigma: float, is_bw: bool) -> Image.Image:
    x_obj_points, y_obj_points = point_selection.get_obj_pixels(source_img_path) # координаты пикселей объекта
    x_bck_points, y_bck_points = point_selection.get_bck_pixels(source_img_path) # координаты пикселей фона

    if ((len(x_obj_points) == 0) or (len(x_bck_points) == 0)):
        print("choose object and background pixel")
        return

    graph, height, width, K = image_graph.get_graph(source_img_path, x_obj_points, y_obj_points, x_bck_points, y_bck_points, _lambda, sigma, is_bw)
    max_flow_value, flow, min_cut = max_flow.maxflow(graph, height * width + 2)
    image = image_graph.get_bwimage(min_cut, width, height)
    image.show()
    return image


if __name__ == "__main__":
    img_input_fp  = "images-320/flower-gr-320.jpg"
    img_output_fp = "our-segments-320/flower-gr-320.png"
    ref_img_fp =  "image-segments-320/flower-gr-320.jpg"

    img_output = get_segmentation(img_input_fp, 100, 1.0, True)
    img_output.save(img_output_fp)

    ref_img = Image.open(ref_img_fp)
    quality_metric.quality_metrics(ref_img, img_output)
