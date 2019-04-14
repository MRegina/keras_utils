import keras
from keras.datasets import cifar10
from keras.models import model_from_json

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# normalize data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

num_classes = 10

#encode ground truth values as one-hot vectors
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

batch_size = 32
epochs = 1

# initiate optimizers
optimizer_A = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optimizer_B = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
optimizer_C = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Compile and train model A
json_file = open('json_files/model_A.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_A = model_from_json(loaded_model_json)
model_A.compile(loss='categorical_crossentropy', optimizer=optimizer_A, metrics=['accuracy'])
model_A.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

# Compile and train model B
json_file = open('json_files/model_B.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_B = model_from_json(loaded_model_json)
model_B.compile(loss='categorical_crossentropy', optimizer=optimizer_B, metrics=['accuracy'])
model_B.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)

# Compile and train model C
json_file = open('json_files/model_C.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_C = model_from_json(loaded_model_json)
model_C.compile(loss='categorical_crossentropy', optimizer=optimizer_C, metrics=['accuracy'])
model_C.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test), shuffle=True)
