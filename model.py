import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Conv2D, Conv3D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model
from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras import backend as k
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping

img_width, img_height,channels = 512,512,3

#Creating layers
img_input = Input(shape=(img_width, img_height,channels))

x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
#VGG Ends here
#Block 5
x = Conv2D(512, (5, 5), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (5, 5), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (5, 5), activation='relu', padding='same', name='block5_conv3')(x)

#Block 6
x = Conv2D(4096, (25, 25), activation='relu', padding='same', name='block6_conv1')(x)
x = Conv2D(4096, (1, 1), activation='relu', padding='same', name = 'block6_conv2')(x)
x = Conv2DTranspose(21, (1, 1), activation='sigmoid', padding='same', name = 'block6_conv3')(x)

#Block 7
x = Conv3D(21,)






def model(input_shape):
