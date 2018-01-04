import matplotlib.pyplot as plt
import tkinter
import pygame
import pygame.camera
import base64
import math
import numpy as np
from PIL import Image
from io import BytesIO
from pygame.locals import *
from sklearn.externals import joblib
from utils import transform_image_to_data_vector

class LiveGestureTester(tkinter.Tk):
  def __init__(self, path_to_model, contain):
    tkinter.Tk.__init__(self)
    pygame.init()
    pygame.camera.init()

    camlist = pygame.camera.list_cameras()
    cam = pygame.camera.Camera(camlist[0],(640,360)) # make sure to preserve 16:9 ratios
    cam.start()

    self.contain = contain
    self.model = LiveGestureTester.load_model(path_to_model)
    self.label = tkinter.Label(text="", compound="top")
    self.label.pack(side="top", padx=8, pady=8)
    self.cam = cam
    self.update_image(delay=1)

  def update_image(self, delay, event=None):
    image_and_label = self.get_image_and_label()
    detected_class = image_and_label[1]
    self.image = image_and_label[0]
    self.label.configure(image=self.image, text=detected_class)
    self.after(delay, self.update_image, delay)

  def get_image_and_label(self):
    output_buffer = BytesIO()

    pix = LiveGestureTester.get_pixels_from_camera_image(self.cam.get_image())
    for row in pix:
      for pixel in row:
        LiveGestureTester.set_black_if_not_skin(pixel)
    # [LiveGestureTester.set_black_if_not_skin(pixel) for pixel in [row for row in pix]]  

    pil_image = Image.fromarray(pix)
    
    pil_image.save(output_buffer, format='gif')
    base_64_data = output_buffer.getvalue()

    data = base64.b64encode(base_64_data).decode()
    image = tkinter.PhotoImage(data=data)
    data_vector = transform_image_to_data_vector(pil_image, self.contain)
    probabilities = list(self.model.predict_proba(data_vector.reshape(1,-1)))
    # print(probabilities)
    probabilities = [round(p,2) for p in probabilities[0]]
    label = str(probabilities) + "\n" + str(self.model.predict(data_vector.reshape(1,-1)))
    # print(label)
    return (image, label)

  def get_pixels_from_camera_image(img_from_camera):
    pix = pygame.surfarray.array3d(img_from_camera)
    pix = pix.swapaxes(0,1) # image from pygame is flipped
    return pix

  def load_model(path_to_model):
    return joblib.load(path_to_model) 

  #WIP
  def set_black_if_not_skin(pixel):
    r, g, b = pixel
    # print(r, g, b)
    color_sum = 1.0 * (r+b+g)
    r /= color_sum
    g /= color_sum
    b /= color_sum
    # print(r, g, b)

    # color_sum = 1.0 * (r+b+g)
    # s = round((.5 * ((r-g)+(r-b)) / math.sqrt((r-g)**2+(r-b)*(g-b))), 2)
    s = (.5 * ((r-g)+(r-b)) / math.sqrt((r-g)**2+(r-b)*(g-b)))

    # print(s)
    h = math.acos(s)
    s = 1 - 3*(min(r, g, b) / color_sum)
    v = color_sum / 3

    y = 0.299 * r + 0.287 * g + 0.11 * b
    cr = r - y
    cb = b - y

    if((0.0 <= h <= 50.0
      and 0.23 <= s <= 0.68
      and r > 95 and g > 40 and b > 20
      and r > g and r > b and abs(r - g) > 15
      #and a > 15)
      )
      or 
      (r > 95 and g > 40
      and b > 20 and r > g and r > b
      and abs(r - g) > 15# and a > 15
      and cr > 135 and cb > 85
      and y > 80 and cr <= (1.5862*cb)+20
      and cr>=(0.3448*cb)+76.2069
      and cr >= (-4.5652*cb)+234.5652
      and cr <= (-1.15*cb)+301.75
      and cr <= (-2.2857*cb)+432.85)):
      print('setting pixel to black')
      pixel[0] = 0
      pixel[1] = 0
      pixel[2] = 0
