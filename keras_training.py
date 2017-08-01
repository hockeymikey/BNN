import os

from keras.engine import Layer
from keras.utils import to_categorical

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import pickle

import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, AveragePooling2D, Reshape, np
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#tf = channels last?
#th = channels first
K.set_image_dim_ordering('tf')

from data_prep.data_utils import roll_data_channel_last


def single_class_accuracy(interesting_class_id):
  def fn(y_true, y_pred):
    class_id_true = K.argmax(y_true, axis=-1)
    class_id_preds = K.argmax(y_pred, axis=-1)
    # Replace class_id_preds with class_id_true for recall here
    accuracy_mask = K.cast(K.equal(class_id_preds, interesting_class_id), 'int32')
    class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
    class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
    return class_acc
  
  return fn

def conv_bn_relu_pool(model, num, c_size, filter):
  for i in range(num):
    model.add(Conv2D(filters=c_size, kernel_size=filter,data_format='channels_last'))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1),data_format='channels_last'))
    
  model.add(keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                                                  beta_initializer='zeros', gamma_initializer='ones',
                                                  moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                                  beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                                  gamma_constraint=None))
  model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                            shared_axes=None))
  model.add(MaxPooling2D(pool_size=(2, 2),strides=2,padding='same',data_format='channels_last'))
  #model.add(AveragePooling2D(pool_size=(2, 2), strides=2, padding='same', data_format='channels_last'))
  
  
def conv_bn_relu(model,num,c_size,filter):
  for i in range(num):
    model.add(Conv2D(filters=c_size, kernel_size=filter,data_format='channels_last'))
    model.add(keras.layers.convolutional.ZeroPadding2D(padding=(1, 1),data_format='channels_last'))
    
  model.add(keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                                                  beta_initializer='zeros', gamma_initializer='ones',
                                                  moving_mean_initializer='zeros', moving_variance_initializer='ones',
                                                  beta_regularizer=None, gamma_regularizer=None, beta_constraint=None,
                                                  gamma_constraint=None))
  #model.add(Reshape((128, 128, 3), input_shape=(160, 320, 3))
  model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                            shared_axes=None))
  
def getdata():
  with open('./cache/new_40x_data_cache_tiny.pkl','rb') as f:
    datav = pickle.load(f)
    #datav = roll_data_channel_last(datav)
    #datav["y_train"][datav["y_train"] > 0] = 1
    #datav["y_val"][datav["y_val"] > 0] = 1
    
    return datav["X_train"],datav["y_train"],datav["X_val"],datav["y_val"]

batch_size = 50
a,s,d,f = getdata()
(x_train, y_train), (x_test, y_test) = (a,s),(d,f)

num_classes = y_train.max()+1
epochs = 30
data_augmentation = False

image_size = x_train.shape[2]

# The data, shuffled and split between train and test sets:

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

metrics = ['accuracy']

for x in np.unique(y_train):
  metrics.append(single_class_accuracy(x))

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()

model.add(Conv2D(image_size*2, (3,3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True,
                                                        beta_initializer='zeros', gamma_initializer='ones',
                                                        moving_mean_initializer='zeros',
                                                        moving_variance_initializer='ones',
                                                        beta_regularizer=None, gamma_regularizer=None,
                                                        beta_constraint=None,
                                                        gamma_constraint=None))
model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                          shared_axes=None))

#conv_bn_relu(model,1,64,(3,3))
conv_bn_relu_pool(model, 3, image_size*4, (3, 3))

#model.add(Dropout(0.22))

conv_bn_relu(model,2,image_size*4,(3,3))
conv_bn_relu(model,2,image_size*4,(3,3))
conv_bn_relu_pool(model,1,image_size*4,(3,3))
conv_bn_relu_pool(model, 2, image_size*4, (3, 3))
conv_bn_relu(model,1,image_size*4,(3,3))

#model.add(Dropout(0.22))

conv_bn_relu(model,1,image_size*4,(1,1))
conv_bn_relu_pool(model, 1, image_size*4, (1, 1))

#model.add(Conv2D(128, (3,3),data_format='channels_last'))

#model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.3))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
'''
conv_bn_relu_pool(model,1,image_size*4,(3,3))
conv_bn_relu_pool(model,1,image_size*2,(3,3))
'''
#model.add(Dropout(0.22))
model.add(Flatten())

model.add(keras.layers.normalization.BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001, center=True, scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, beta_constraint=None, gamma_constraint=None))
#model.add(Activation('relu'))
model.add(keras.layers.advanced_activations.PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None,
                                          shared_axes=None))
model.add(Dense(num_classes))
model.add(Activation('softmax'))#softmas

# initiate RMSprop optimizer
#opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
opt = keras.optimizers.Nadam(lr=4.0e-2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)#0.004
#opt = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=metrics)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        steps_per_epoch=x_train.shape[0] // batch_size,
                        epochs=epochs,
validation_data=(x_test, y_test))
wait = input('wait?')