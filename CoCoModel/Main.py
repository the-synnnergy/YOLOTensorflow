import CoCoData
import os
import pathlib
import YOLOModel
import Util
import Config

Util.fix_gpu()


if Config.loadModel:
    YOLOModel.loadModel()
    
else:
    YOLOModel.createModel()


train_dataset, val_dataset = CoCoData.extract_coco_data()
YOLOModel.train(train_dataset, val_dataset)
YOLOModel.savemodel()
