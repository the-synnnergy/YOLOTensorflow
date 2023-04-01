from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import os

def fix_gpu():
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
def getBasePath():
    current_path = os.getcwd()
    return  os.path.join(current_path, 'cocoDataset')

def bbox_confidence_index(bbox_int):
  return 4+5*bbox_int

def classes_begin(bbox_amount):
  return 5*bbox_amount

def get_bbox_from_index(bbox_index,class_box_1d_tensor):
  return class_box_1d_tensor[bbox_index*5:bbox_index*5+5]

def get_highest_prob_bbox(grid_output,bbox_amount):
  max_index = 0
  for bbox in range(bbox_amount):
    index = bbox_confidence_index(bbox)
    if grid_output[index] > grid_output[max_index]:
      max_index = index
  return max_index

