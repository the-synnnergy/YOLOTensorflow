import json
from PIL import Image
import Config
import os
#import pandas as pd
from pycocotools.coco import COCO
from keras.preprocessing.image import ImageDataGenerator
import sys
import tensorflow as tf
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import Util
import random

# just for debug
def printImages():
    data_dir  = Util.getBasePath()

    ann_file = os.path.join(data_dir,'annotations' , 'instances_train2017.json')
    train_image_dir = os.path.join(data_dir, 'train2017')
    coco = COCO(ann_file)

    # get all image ids in the dataset
    img_ids = coco.getImgIds()

    # loop through each image id and get its annotations
    for img_id in img_ids:
        # load the image using its file name
        img_info = coco.loadImgs(img_id)[0]
        img_file = train_image_dir + '\\' + img_info['file_name']
        image = Image.open(img_file)
        
        # get the annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        # loop through each annotation and extract the label and bounding box
        for ann in anns:
            label = coco.loadCats(ann['category_id'])[0]['name']
            bbox = ann['bbox']
            
            # do something with the label, bbox, and image (e.g. display it)
            print("Label:", label)
            print("BBox:", bbox)
            image.show()

def create_dataset(ann_file, image_dir):
    coco = COCO(ann_file)
    if Config.importMaxImages:
        b = random.randint(0, 2000)
        image_ids = coco.getImgIds()[b:b+Config.maxImgaes]
    else:
        image_ids = coco.getImgIds()
    image_paths = [coco.loadImgs(img_id)[0]['file_name'] for img_id in image_ids]
    
    

    def map_fn(image_path):
        ann_ids = coco.getAnnIds(imgIds=image_ids, iscrowd=None)
        annotations = coco.loadAnns(ann_ids)



        image = tf.io.read_file(image_dir + '/' + image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        shape = tf.shape(image)
        width, height,  channels = shape[0], shape[1], shape[2]
        width, height = 640, 480

        grid_length =  (Config.IMG_SIZE / Config.gridsize ,Config.IMG_SIZE / Config.gridsize)
        oldgrid_length =  (width // Config.gridsize ,height // Config.gridsize)
        bboxes = []
        Picturescalediff = (width/Config.IMG_SIZE,height/Config.IMG_SIZE)

        image = tf.image.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE))
        

        predictions = tf.zeros((Config.gridsize,Config.gridsize,5*Config.boxes+ Config.num_classes),dtype=tf.float32)
        for annotation in annotations:

            bbox = annotation['bbox']
            x, y, w, h = bbox    

            x_cell = x // oldgrid_length[0]
            y_cell = y // oldgrid_length[1]

            #some boxes are out of bound
            if x_cell < Config.gridsize and y_cell < Config.gridsize:
                x, y, w, h = x/Picturescalediff[0], y/Picturescalediff[1], w/Picturescalediff[0], h/Picturescalediff[1]

                #x = tf.cast(x, tf.float32)
                #y = tf.cast(y, tf.float32)
                #w = tf.cast(w, tf.float32)
                #h = tf.cast(h, tf.float32)
                
                x_cell = tf.cast(x_cell, tf.float32)
                y_cell = tf.cast(y_cell, tf.float32)
                grid_length = tf.cast(grid_length, tf.float32)

                xPos = x - tf.cast(x_cell*grid_length[0], dtype=tf.float32)
                yPos = y - tf.cast(y_cell*grid_length[1], dtype=tf.float32)


                c = tf.stack([w, h, 1.0],axis=0)
                bboxes_tensor = tf.stack([xPos, yPos], axis=0)
                bboxes_tensor = tf.reshape(bboxes_tensor, [2]) 
                bboxes_tensor = tf.concat([bboxes_tensor, c], axis=0)

                bboxes_tensor = tf.tile(tf.expand_dims(bboxes_tensor, axis=1), multiples=[1, Config.boxes])
                bboxes_tensor = tf.reshape(bboxes_tensor, shape=[-1])
                label = tf.one_hot(annotation['category_id'], depth=Config.num_classes, dtype=tf.float32) 
                obj_pred = tf.concat([ bboxes_tensor, label], axis=0)

                predictions = tf.tensor_scatter_nd_update(predictions, indices=[[int(x_cell), int(y_cell), i] for i in range(Config.num_classes+Config.boxes*5)], updates=obj_pred)


        return image,predictions

            

        
    
    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.shuffle(buffer_size=Config.buffer_size)
    print("preprocess Data")
    #
    # dataset = dataset.map(lambda x: tf.py_function(map_fn, [x], [tf.float32]), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(Config.batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset


def setnumOfClases(ann_file):
    coco = COCO(ann_file)
    #Config.num_classes = len(coco.getCatIds())

def extract_coco_data():
    data_dir  = Util.getBasePath()
    # Load the COCO dataset annotations
    train_ann_file = os.path.join(data_dir,'annotations' , 'instances_train2017.json')
    train_image_dir = os.path.join(data_dir, 'train2017')

    val_ann_file = os.path.join(data_dir,'annotations' , 'instances_val2017.json')
    val_image_dir = os.path.join(data_dir, 'val2017')

    
    train_dataset = create_dataset(train_ann_file,train_image_dir)
    val_dataset = create_dataset(val_ann_file,val_image_dir)



    return train_dataset, val_dataset