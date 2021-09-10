import torch
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

def show_img(img, vmin, vmax, cmap='gray'):
  plt.figure(figsize=(16,10))
  plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
  plt.axis(False)
  plt.show()

def transform_img(img, transform):
  img = transform(img).unsqueeze(0)
  assert img.shape[-2] <= 1600 and img.shape[-1] <= 1600, 'demo model only supports images up to 1600 pixels on each side'
  return img