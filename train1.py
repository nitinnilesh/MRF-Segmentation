#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import shutil

import click
import numpy as np
from keras import callbacks, optimizers
from IPython import embed

from model_final import *
from utils.image_reader import (
    RandomTransformer,
    SegmentationDataGenerator)
from keras.utils.multi_gpu_utils import multi_gpu_model
import keras.backend as K
import tensorflow as tf
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

@click.command()
@click.option('--train-list-fname', type=click.Path(exists=True),
              default='./VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt')
@click.option('--val-list-fname', type=click.Path(exists=True),
              default='./VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')
@click.option('--img-root', type=click.Path(exists=True),
              default='./VOCdevkit/VOC2012/JPEGImages')
@click.option('--mask-root', type=click.Path(exists=True),
              default='./VOCdevkit/VOC2012/SegmentationClass')
@click.option('--batch-size', type=int, default=1)
@click.option('--learning-rate', type=float, default=1e-7)
def train(train_list_fname,
          val_list_fname,
          img_root,
          mask_root,
          batch_size,
          learning_rate):

    # Create image generators for the training and validation sets. Validation has
    # no data augmentation.
    transformer_train = RandomTransformer(horizontal_flip=True, vertical_flip=True)
    datagen_train = SegmentationDataGenerator(transformer_train)

    transformer_val = RandomTransformer(horizontal_flip=False, vertical_flip=False)
    datagen_val = SegmentationDataGenerator(transformer_val)

    '''train_desc = '{}-lr{:.0e}-bs{:03d}'.format(
        time.strftime("%Y-%m-%d %H:%M"),
        learning_rate,
        batch_size)
    checkpoints_folder = 'trained/' + train_desc
    try:
        os.makedirs(checkpoints_folder)
    except OSError:
        shutil.rmtree(checkpoints_folder, ignore_errors=True)
        os.makedirs(checkpoints_folder)

    model_checkpoint = callbacks.ModelCheckpoint(
        checkpoints_folder + '/ep{epoch:02d}-vl{val_loss:.4f}.hdf5',
        monitor='loss')'''


    '''tensorboard_cback = callbacks.TensorBoard(
        log_dir='{}/tboard'.format(checkpoints_folder),
        histogram_freq=0,
        write_graph=False,
        write_images=False)
    csv_log_cback = callbacks.CSVLogger(
        '{}/history.log'.format(checkpoints_folder))
    reduce_lr_cback = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        verbose=1,
        min_lr=0.05 * learning_rate)'''

    model = DPN((224,224,3))
    #gpu_model = multi_gpu_model(model, gpus=2)
    def nll(y_true,y_pred):
        loss = K.sum(K.binary_crossentropy(y_true,y_pred),axis=-1)
        return loss
    def cross_entropy_2d(target,input):
        input = K.eval(input)
        input = Variable(torch.from_numpy(input).type(torch.FloatTensor))
        target = K.eval(target)
        target = Variable(torch.from_numpy(target).type(torch.LongTensor))
        
        n, h, w, c = input.size()
        log_p = F.log_softmax(input, dim=1)
        log_p = log_p.contiguous().view(-1, c)
        log_p = log_p[target.view(n * h * w, 1).repeat(1, c) >= 0]
        log_p = log_p.view(-1, c)

        mask = target >= 0
        target = target[mask]
        loss = F.nll_loss(log_p, target, ignore_index=250,weight=weight, size_average=False)
        loss /= mask.data.sum()
        return loss.data[0]
    model.compile(loss=nll,optimizer=optimizers.Adam(lr=1e-8),metrics=['accuracy'])
    # Build absolute image paths
    def build_abs_paths(basenames):
        img_fnames = [os.path.join(img_root, f) + '.jpg' for f in basenames]
        mask_fnames = [os.path.join(mask_root, f) + '.png' for f in basenames]
        return img_fnames, mask_fnames

    train_basenames = [l.strip() for l in open(train_list_fname).readlines()]
    val_basenames = [l.strip() for l in open(val_list_fname).readlines()][:512]

    train_img_fnames, train_mask_fnames = build_abs_paths(train_basenames)
    val_img_fnames, val_mask_fnames = build_abs_paths(val_basenames)

    '''skipped_report_cback = callbacks.LambdaCallback(
        on_epoch_end=lambda a, b: open(
            '{}/skipped.txt'.format(checkpoints_folder), 'a').write(
            '{}\n'.format(datagen_train.skipped_count)))'''

    model.fit_generator(
        datagen_train.flow_from_list(
            train_img_fnames,
            train_mask_fnames,
            shuffle=True,
            batch_size=8,
            img_target_size=(224, 224),
            mask_target_size=(224, 224)),
        steps_per_epoch=len(train_basenames)//8,
        epochs=50,
        validation_data=datagen_val.flow_from_list(
            val_img_fnames,
            val_mask_fnames,
            batch_size=8,
            img_target_size=(224, 224),
            mask_target_size=(224, 224)),validation_steps=len(val_basenames)//8)
    model.save('DPN_Adam_50ep_1e_8_wodec_womgpu.h5')


if __name__ == '__main__':
    train()
