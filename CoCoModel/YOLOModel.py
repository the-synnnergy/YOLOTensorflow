import Config
import tensorflow as tf
from keras import backend as K
import numpy as np
from keras.models import load_model
from keras.optimizers import Adam
import Util
from tensorflow import keras
currentModel = None

class YOLOModel(tf.keras.Model):

    def __init__(self,optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.base_model = tf.keras.applications.EfficientNetB0(include_top = False,weights="imagenet",input_shape=(Config.IMG_SIZE,Config.IMG_SIZE,3))
        self.base_model.trainable = False
        self.dense1 = tf.keras.layers.Dense(Config.DENSE1_SIZE)
        self.leaky1 = tf.keras.layers.LeakyReLU(alpha=0.1)
        self.output_size = Config.gridsize*Config.gridsize *(Config.boxes*5+ Config.num_classes)
        self.dense2 = tf.keras.layers.Dense(Config.boxes*5+ Config.num_classes)

    def call(self,inputs):
        x = self.base_model(inputs)
        x = self.dense1(x)
        x = self.leaky1(x)
        x = self.dense2(x)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({'optimizer': self.optimizer})
        return config

    @classmethod
    def from_config(cls, config):
        optimizer = config.pop('optimizer')
        model = cls(optimizer=optimizer)

        return model

def get_iou_for_boxes(true_boxes, pred_boxes):
  # Calculate the IoU
  x1 = tf.maximum(true_boxes[..., 0], pred_boxes[..., 0])
  y1 = tf.maximum(true_boxes[..., 1], pred_boxes[..., 1])
  x2 = tf.minimum(true_boxes[..., 2], pred_boxes[..., 2])
  y2 = tf.minimum(true_boxes[..., 3], pred_boxes[..., 3])

  intersection_area = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

  box1_area = (true_boxes[..., 2] - true_boxes[..., 0]) * (true_boxes[..., 3] - true_boxes[..., 1])
  box2_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])

  union_area = box1_area + box2_area - intersection_area

  iou = intersection_area / union_area

  return iou


#returns the boxes from y_true and y_pred in the shape [batch_size, gridsize, gridsize, 5]
# chooses with the highest iou
def CondenceBoxesIou(y_true, y_pred):
    
    class_begin_idx = Util.classes_begin(Config.boxes)

    pred_boxes = y_pred[...,:class_begin_idx]
    pred_boxes = tf.reshape(pred_boxes,[Config.batch_size, Config.gridsize, Config.gridsize, Config.boxes, 5])

    true_boxes = y_true[...,:class_begin_idx]
    true_boxes = tf.reshape(true_boxes,[Config.batch_size, Config.gridsize, Config.gridsize, Config.boxes, 5])


    #calculate Iou
    iou = get_iou_for_boxes(true_boxes,pred_boxes)

    

    max_idx = tf.math.argmax(iou, axis=-1)



    newpred_boxes = tf.gather(pred_boxes,max_idx,batch_dims=3)
    newtrue_boxes = tf.gather(true_boxes,max_idx,batch_dims=3)

    newpred_boxes = tf.reshape(newpred_boxes,(Config.batch_size, Config.gridsize, Config.gridsize, 5))
    newtrue_boxes = tf.reshape(newtrue_boxes,(Config.batch_size, Config.gridsize, Config.gridsize, 5))

    return newtrue_boxes, newpred_boxes

#y_true and y_pred have the shape [batch_size, gridsize, gridsize, 5*num_boxes+num_classes]
# every box in the same dim in y_true has the sam values
class YOLOLoss(tf.keras.losses.Loss):
  def call(self, y_true, y_pred):
    true_boxes,pred_boxes = CondenceBoxesIou(y_true,y_pred)


    class_begin_idx = Util.classes_begin(Config.boxes)

    true_classes = y_true[...,class_begin_idx:]
    pred_classes = y_pred[...,class_begin_idx:]

    active_grids =   tf.cast(tf.math.count_nonzero(true_classes,axis=-1), dtype=tf.float32)
    active_grids =  tf.where(active_grids!= 0, tf.ones_like(active_grids), active_grids)


    xy_loss =  Config.lambda_coord * K.sum(K.sum(tf.math.multiply(active_grids, K.square(pred_boxes[...,0]-true_boxes[...,0])+K.square(pred_boxes[...,1]-true_boxes[...,1])),axis=-1),axis=-1)
    wh_loss =  Config.lambda_coord * K.sum(K.sum(tf.math.multiply(active_grids, K.square(K.sqrt(pred_boxes[...,2])-K.sqrt(true_boxes[...,2]))+K.square(K.sqrt(pred_boxes[...,3])-K.sqrt(true_boxes[...,3]))),axis=-1),axis=-1)  


    confidence_loss = K.sum(K.sum(tf.math.multiply(active_grids,K.square(pred_boxes[...,4]- true_boxes[...,4])),axis=-1),axis=-1)
    inverted_active_grids = tf.math.logical_not(tf.cast(active_grids,dtype=tf.bool))
    inverted_active_grids = tf.cast(inverted_active_grids,dtype=tf.float32)
    noob_confidence_loss = Config.lambda_noobj *K.sum(K.sum(tf.math.multiply(inverted_active_grids,K.square(pred_boxes[...,4]- true_boxes[...,4])),axis=-1),axis=-1)
    class_prob_loss = K.sum(K.sum(tf.math.multiply(active_grids,K.sum(K.square(pred_classes-true_classes),axis=-1)),axis=-1),axis=-1)


    return xy_loss + wh_loss + confidence_loss + noob_confidence_loss + class_prob_loss
  

  


def createModel():
    print("create Model")
    optimizer = Adam(learning_rate=0.001)
    YOLOModel.currentModel = YOLOModel(optimizer)

    YOLOModel.currentModel.compile(optimizer='adam',
              loss=YOLOLoss(),
              run_eagerly= True,
              metrics=['accuracy'])
    if YOLOModel.currentModel is None:
        print("Error: Can't create Model")
        exit()
    

    print("created Model")


def loadModel():
    print("loading Model")
    adam = Adam(learning_rate=0.001)
    custom_objects = {'Adam': adam}
    YOLOModel.currentModel = keras.models.load_model('YOLOModel.keras', custom_objects=custom_objects)
    YOLOModel.currentModel.compile(optimizer='adam',
              loss=YOLOLoss(),
              run_eagerly= True,
              metrics=['accuracy'])

    if YOLOModel.currentModel is None:
        print("Error: Can't load Model")
        exit()
    
    print("loaded Model")


def savemodel():
    print("save Model")
    YOLOModel.currentModel.save('YOLOModel.keras', include_optimizer=True)

    
def call(input):
    if YOLOModel.currentModel is None:
        loadModel()
    return YOLOModel.currentModel.call(input)


def train(train_dataset, val_dataset):
    if YOLOModel.currentModel is None:
        loadModel()

    print("train Model")
    YOLOModel.currentModel.fit(train_dataset, epochs=5, validation_data=val_dataset)
