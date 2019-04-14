import keras
from keras.models import model_from_json
import json
from copy import deepcopy

# Combine_models function creates one model json string from two keras model jsons. The combined model inherits layers
# from model A (layers before a specific layer named layername_A) and inherits its last layers from model B (layers
# following a layer named layer_B and including layername_B)
def combine_models(json_A, json_B, layername_A, layername_B):
    # json_A: path to json file of model A (from which the input part of the combined model comes)
    # json_B: path to json file of model B (from which the output part of the combined model comes)
    # layername_A: name of a layer in model A, that will be replaced with layername_B from model B
    # layername_B: the first layer that will be used from model B, to replace layername_A from model A
    with open(json_A) as json_file:
        model_A_json = json.load(json_file)

    with open(json_B) as json_file:
        model_B_json = json.load(json_file)

    for i, layer in enumerate(model_A_json['config']['layers']):
        if layer['name'] == layername_A:
            idx_A=i

    for i, layer in enumerate(model_B_json['config']['layers']):
        if layer['name'] == layername_B:
            idx_B=i

    model_C_json = deepcopy(model_A_json)
    model_C_json['config']['layers'] = model_A_json['config']['layers'][:idx_A] + model_B_json['config']['layers'][idx_B:]
    model_C_json['config']['layers'][idx_A]['inbound_nodes'] = model_A_json['config']['layers'][idx_A]['inbound_nodes']
    model_C_json['config']['output_layers'] = model_B_json['config']['output_layers']

    model_C_json_string = json.dumps(model_C_json)
    return model_C_json_string

model_C_json_string = combine_models('json_files/model_A.json', 'json_files/model_B.json', layername_A ='global_average_pooling2d_1', layername_B='Conv1_B')

model_C = model_from_json(model_C_json_string)
model_C_json = model_C.to_json()
with open("json_files/model_C.json", "w") as json_file:
    json_file.write(model_C_json)

