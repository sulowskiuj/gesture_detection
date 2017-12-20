import numpy as np
import os, sys, random, math
from PIL import Image
from tqdm import tqdm

def generate_flipped_videos(path_to_video_container='videos'):
  for root, dirs, files in os.walk(path_to_video_container):
    for filename in files:
      if filename.startswith("flipped"):
        continue
      path_to_file = os.path.join(root, filename)
      path_to_flipped_file = os.path.join(root, "flipped_" + filename)
      os.system("ffmpeg -n -i " + path_to_file + " -vf hflip -c:a copy " + path_to_flipped_file)

def extract_images_from_videos(path_to_video_container='videos', path_to_image_container='images'):
  subdirectories = next(os.walk(path_to_video_container))[1]
  for subdirectory_name in subdirectories:
    subdirectory_path = os.path.join(path_to_video_container, subdirectory_name)
    image_subdirectory_path = os.path.join(path_to_image_container, subdirectory_name)
    
    if not os.path.exists(image_subdirectory_path):
      os.makedirs(image_subdirectory_path)

    for video_name in os.listdir(subdirectory_path):
      video_path = os.path.join(subdirectory_path, video_name)
      image_path = os.path.join(image_subdirectory_path, os.path.splitext(video_name)[0])
      os.system("ffmpeg -i " + video_path + " -vf fps=30 -qscale:v 2 " + image_path + "%d.jpg")

def generate_dataset(path_to_image_container='images', path_do_dataset_container='datasets', contain=32, flatten=False, grayscale=False):
  subdirectories = next(os.walk(path_to_image_container))[1]
  number_of_images = sum([len(files) for r, d, files in os.walk(path_to_image_container)])

  data   = []
  labels = []
  number_of_subdirectories = len(subdirectories)
  i = 1

  for subdirectory_name in subdirectories:
    subdirectory_path = os.path.join(path_to_image_container, subdirectory_name)

    print('subdirectory {0} of {1}'.format(i, number_of_subdirectories))
    i += 1
    
    for image_name in tqdm(os.listdir(subdirectory_path)):
      image_path = os.path.join(subdirectory_path, image_name)
      image = Image.open(image_path)
      # image = transform_image_to_data(image, contain)

      vector = transform_image_to_data_vector(image, contain, grayscale, flatten)
      data.append(vector)
      labels.append(subdirectory_name)

  file_suffix = ''
  if(grayscale):
    file_suffix += 'grayscale_'
  if(flatten):
    file_suffix += 'flattened_'

  np.save(os.path.join(path_do_dataset_container, 'data_{0}{1}'.format(file_suffix, contain)), np.array(data))
  np.save(os.path.join(path_do_dataset_container, 'labels_{0}{1}'.format(file_suffix, contain)), np.array(labels))

def transform_image_to_data_vector(image, contain, grayscale, flatten):
  width, height = image.size
  if(height > width):
    width, height = height, width
    image = image.rotate(90, expand=True)
  ratio = 1.0 * (width / height) / (16.0/9)
  new_height = math.ceil(height * ratio)
  # print('ratio should be ' + str(round(16.0/9, 2)) + ' and is ' + str(round(width / new_height, 2)))
  area = (0, 0, width, new_height)
  image = image.crop(area)
  image.thumbnail((contain, contain), Image.ANTIALIAS)
  if(grayscale):
    image = image.convert('L')
  pixels = np.array(image, dtype=np.uint8)
  if(flatten):
    vector = pixels.flatten()
  else:
    vector = pixels
  return vector
