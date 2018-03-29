import numpy as np
import os
import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Activation, Flatten
from keras.layers import Conv2D, Conv3D, MaxPooling2D, UpSampling2D, Conv2DTranspose
from keras.models import Model
from keras import backend as k
import h5py
from keras.applications.vgg16 import VGG16
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def DPN(input_shape):
	#Take VGG first 4 blocks, where last block without pool layer
	vgg16 = VGG16(weights='imagenet',include_top=False, input_shape=(512,512,3))
	#Block 5
	x = Conv2D(512, (5, 5), activation='relu', padding='same', name='block5_conv1')(vgg16.layers[-6].output)
	x = Conv2D(512, (5, 5), activation='relu', padding='same', name='block5_conv2')(x)
	x = Conv2D(512, (5, 5), activation='relu', padding='same', name='block5_conv3')(x)
	#Block 6
	x = Conv2D(4096, (25, 25), activation='relu', padding='same', name='block6_conv1')(x)
	x = Conv2D(4096, (1, 1), activation='relu', padding='same', name = 'block6_conv2')(x)
	x = Conv2D(21, (1, 1), activation='sigmoid', padding='same', name = 'block6_conv3')(x)
	x = UpSampling2D((8,8))(x)
	# Model creted for Unary terms

	model = Model(inputs=vgg16.input, outputs=x)
	weights_path = 'vgg16_weights.h5'
	change_weights_conv(model,vgg16,14,15)
	change_weights_conv(model,vgg16,15,16)
	change_weights_conv(model,vgg16,16,17)
	print('Conv layers done')
	change_weights_fc1(model, 17, weights_path, obj=[1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6], old_dim = (7,7,512,4096), new_dim = (25,25,512,4096))
	print('FC1 Done')
	change_weights_fc2(model, 18, weights_path, new_dim = (1,1,4096,4096))
	print('FC2 Done')
	return model

def change_weights_conv(model,vgg16,m_layer_number,v_layer_number,obj = [1,2], new_dim = (5,5,512,512)):
	weights = vgg16.layers[v_layer_number].get_weights()[0]
	bias = vgg16.layers[v_layer_number].get_weights()[1]
	weights_modified = np.zeros(new_dim)
	for i in range(weights.shape[2]):
		for j in range(weights.shape[3]):		
			temp = np.insert(weights[:,:,i,j],obj,0,axis=1)
			temp = np.insert(temp.T,obj,0,axis=1).T
			weights_modified[:,:,i,j] = temp
	new_weights = [weights_modified,bias]
	model.layers[m_layer_number].set_weights(new_weights)

def change_weights_fc1(model, m_layer_number, weights_path, obj, old_dim, new_dim):
	f = h5py.File(weights_path)
	weights = np.asarray(f['fc1']['fc1_W_1:0'])
	bias = np.asarray(f['fc1']['fc1_b_1:0'])
	weights = weights.reshape(7,7,512,4096)
	weights_modified = np.zeros(new_dim)
	for i in range(weights.shape[2]):
		for j in range(weights.shape[3]):
			temp = np.insert(weights[:,:,i,j],obj,0,axis=1)
			temp = np.insert(temp.T,obj,0,axis=1).T
			weights_modified[:,:,i,j] = temp
	new_weights = [weights_modified,bias]
	model.layers[m_layer_number].set_weights(new_weights)

def change_weights_fc2(model, m_layer_number, weights_path, new_dim):
	f = h5py.File(weights_path)
	weights = np.asarray(f['fc2']['fc2_W_1:0'])
	bias = np.asarray(f['fc2']['fc2_b_1:0'])
	weights_modified = np.zeros(new_dim)
	for i in range(weights.shape[0]):
		for j in range(weights.shape[1]):
			weights_modified[:,:,i,j] = weights[i,j]
	new_weights = [weights_modified,bias]
	model.layers[m_layer_number].set_weights(new_weights)

def main():
	model = DPN((512,512,3))
	model.summary()
	print('done')

if __name__=='__main__':
	main()
