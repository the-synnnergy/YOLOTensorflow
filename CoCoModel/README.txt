To train and validate with CoCo the dataset needs to be in the same folder in the following Layout


cocoDataset/
├── annotations/
│    ├── instances_train2017.json
│    ├── instances_val2017.json
├── train2017/
│    ├── 000000000009.jpg
│    ├── ...
├── val2017/
     ├── 000000000776.jpg
     ├── ...


Running the current Main results in loading/creating a Model, training it with the CoCo Dataset and saving the Model in 'YOLOModel.keras'.