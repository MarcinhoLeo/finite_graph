import tkinter as tk
from PIL import ImageTk, Image
import os

obj_pixel_x = []
obj_pixel_y = []
bck_pixel_x = []
bck_pixel_y = []

def get_obj_pixels(filepath):
    global obj_pixel_x, obj_pixel_y
    root = tk.Tk()
    root.geometry("+585+300")
    img = ImageTk.PhotoImage(Image.open(filepath))
    panel = tk.Label(root, image = img)
    panel.pack(anchor = 's')
    panel.bind('<Button-1>', onclick_obj)
    root.mainloop()
    print('in object we have', obj_pixel_x, obj_pixel_y)

    ret_obj_pixel_x = obj_pixel_x.copy()
    ret_obj_pixel_y = obj_pixel_y.copy()
    obj_pixel_x = []
    obj_pixel_y = []
    return ret_obj_pixel_x, ret_obj_pixel_y

def get_bck_pixels(filepath):
    global bck_pixel_x, bck_pixel_y
    root = tk.Tk()
    root.geometry("+585+300")
    img = ImageTk.PhotoImage(Image.open(filepath))
    panel = tk.Label(root, image = img)
    panel.pack(anchor = 's')
    panel.bind('<Button-1>', onclick_bck)
    root.mainloop()
    print('in back we have', bck_pixel_x, bck_pixel_y)

    ret_bck_pixel_x = bck_pixel_x.copy()
    ret_bck_pixel_y = bck_pixel_y.copy()
    bck_pixel_x = []
    bck_pixel_y = []
    return ret_bck_pixel_x, ret_bck_pixel_y

def onclick_obj(event):
    global obj_pixel_x, obj_pixel_y
    obj_pixel_x.append(event.x)
    obj_pixel_y.append(event.y)
    print(event.x, event.y)

def onclick_bck(event):
    global bck_pixel_x, bck_pixel_y
    bck_pixel_x.append(event.x)
    bck_pixel_y.append(event.y)
    print(event.x, event.y)
