from keras.models import load_model
import numpy as np
import os
from PIL import Image
from utils import pascal_palette, interp_map
from utils import image_reader as ir
import scipy.misc as misc
import matplotlib.pyplot as plt
import keras.backend as K
from keras.preprocessing.image import load_img, img_to_array,flip_axis

input_width, input_height = 224, 224
label_margin = 186
pascal_mean = np.array([102.93, 111.36, 116.52])
input_image_path = '2009_004436.jpg'

def add_context_margin(image, margin_size, **pad_kwargs):
    """ Adds a margin-size border around the image, used for
    providing context. """
    return np.pad(image,
                  ((margin_size, margin_size),
                   (margin_size, margin_size),
                   (0, 0)), **pad_kwargs)

def pad_to_square(image, min_size, **pad_kwargs):
    """ Add padding to make sure that the image is larger than (min_size * min_size).
    This time, the image is aligned to the top left corner. """

    h, w = image.shape[:2]

    if h >= min_size and w >= min_size:
        return image

    top = bottom = left = right = 0

    if h < min_size:
        top = (min_size - h) // 2
        bottom = min_size - h - top
    if w < min_size:
        left = (min_size - w) // 2
        right = min_size - w - left

    return np.pad(image,
                  ((top, bottom),
                   (left, right),
                   (0, 0)), **pad_kwargs)

def pad_image(image):
    image_pad_kwargs = dict(mode='reflect')
    image = add_context_margin(image, label_margin, **image_pad_kwargs)
    return pad_to_square(image, 224, **image_pad_kwargs)

def crop_to(image, target_h=224, target_w=224):
    h_off = (image.shape[0] - target_h) // 2
    w_off = (image.shape[1] - target_w) // 2
    return image[h_off:h_off + target_h,
           w_off:w_off + target_w, :]

def rgb_to_bgr(image):
    # Swap color channels to use pretrained VGG weights
    return image[:, :, ::-1]

def remove_mean(image):
    # Note that there's no 0..1 normalization in VGG
    return image - pascal_mean

def nll(y_true,y_pred):
    import keras.backend as K
    return K.sum(K.binary_crossentropy(y_true,y_pred))

def get_trained_model():
    model_name = 'DPN_Adam_50ep_1e_8_wodec_womgpu.h5'
    model = load_model(model_name,custom_objects={'nll': nll})
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
    rgb[:, :, 0] = r#/255.0 *100.0
    rgb[:, :, 1] = g#/255.0*100.0
    rgb[:, :, 2] = b#/255.0*100.0
    #plt.imshow(rgb)
    #plt.show()
    return rgb

def forward_pass():
    import scipy.io as sio
    model = get_trained_model()
    image = misc.imread(input_image_path)
    image = misc.imresize(image,(224,224,3),interp='nearest')
    #image = pad_image(image)
    #image = crop_to(image)
    #image = rgb_to_bgr(image)
    #image = remove_mean(image)
    #image = image.astype(np.float64)
    image = image.astype(float)/255.0

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
    prob = ((model.predict(net_in)[0]))#K.eval, K.softmax
    prob_edge = int(prob.shape[0])
    prob = prob.reshape((input_height,input_width))
    seg_image = decode_segmap(prob)#.astype(np.uint8)
    #misc.imsave('result_straight.png',seg_image)
    #sio.savemat('result_raw.mat',mdict={'seg_image':seg_image})
    plt.imshow(seg_image)
    plt.show()
    #print(prob)

if __name__ == '__main__':
    forward_pass()
