import keras
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Input
from keras.models import Model

num_classes = 10

# define model A
input_img_A = Input(shape=(32, 32, 3), name='Input_A')
x = Conv2D(8, (3, 3), padding='same', activation='relu')(input_img_A)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(16, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = GlobalAveragePooling2D()(x)
output_A = Dense(num_classes, activation='softmax')(x)

model_A = Model(inputs=input_img_A, outputs=output_A)

# serialize model A to JSON
model_A_json = model_A.to_json()
with open("json_files/model_A.json", "w") as json_file:
    json_file.write(model_A_json)

#define model B
input_img_B= Input(shape=(32, 32, 3), name='Input_B')
x = Conv2D(8, (3, 3), padding='same', activation='relu', name='Conv1_B')(input_img_B)
x = MaxPooling2D((2, 2), name='MaxPool1_B')(x)
x = Conv2D(16, (3, 3), padding='same', activation='relu', name='Conv2_B')(x)
x = GlobalAveragePooling2D()(x)
output_B = Dense(num_classes,activation='softmax')(x)

model_B = Model(inputs=input_img_B, outputs=output_B)

# serialize model B to JSON
model_B_json = model_B.to_json()
with open("json_files/model_B.json", "w") as json_file:
    json_file.write(model_B_json)

