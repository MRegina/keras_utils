# keras_utils

combine_models:

combine_models.py contains the combine_models function. With this function one can merge two keras models easily, e.g. if you need to add some layers before a specific model's input,
and you do not want to recreate the whole model. The combine_models function reads in two keras model jsons, and generates a third merged 
json file that contains input and layers from model A (before a named layer called layername_A) and layers and output from model B 
(including, and after a layer named layername_B)

A toy example with two extremely simple convolutional models is included in create_models.py and the models can be trained on CIFAR10 with 
train_models.py
