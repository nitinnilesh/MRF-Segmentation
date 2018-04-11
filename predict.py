from keras.models import load_model
import numpy as np
import os
from PIL import Image
from utils import pascal_palette, interp_map
import scipy.misc as misc
import matplotlib.pyplot as plt
import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array,flip_axis

input_width, input_height = 224, 224
label_margin = 186
pascal_mean = np.array([102.93, 111.36, 116.52])
input_image_path = '2007_000033.jpg'

def get_trained_model():
    model_name = 'DPN_Adam_12ep_1e_8_wodec.h5'
    model = load_model(model_name)
    return model

def decode_segmap(label_mask):
    label_colours = pascal_palette
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, 21):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r
    rgb[:, :, 1] = g
    rgb[:, :, 2] = b
    plt.imshow(rgb)
    plt.show()
    return rgb

def forward_pass():
    model = get_trained_model()
    #input_image = np.array(Image.open(input_image_path)).astype(np.float32)
    input_image = misc.imread(input_image_path).astype(np.float32)
    input_image = misc.imresize(input_image,(input_height,input_width,3),interp='bicubic')
    input_image = input_image.astype(float) / 255.0
    image = input_image[:, :, ::-1] - pascal_mean
    image_size = image.shape

    net_in = np.zeros((1, input_height, input_width, 3), dtype=np.float32)
    '''output_height = input_height - 2 * label_margin
    output_width = input_width - 2 * label_margin
    image = np.pad(image,((label_margin, label_margin),
                    (label_margin, label_margin),
                    (0, 0)), 'reflect')

    margins_h = (0, input_height - image.shape[0])
    margins_w = (0, input_width - image.shape[1])
    image = np.pad(image,
                   (margins_h,
                    margins_w,
                    (0, 0)), 'reflect')'''

    # Run inference
    net_in[0] = image
    prob = K.eval(K.softmax(model.predict(net_in)[0]))
    prob_edge = int(prob.shape[0])
    prob = prob.reshape((input_height,input_width))
    seg_image = decode_segmap(prob).astype(np.uint8)
    misc.imsave('result.png',seg_image)
    #print(prob)

if __name__ == '__main__':
    forward_pass()
