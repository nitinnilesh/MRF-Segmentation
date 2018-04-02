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
@click.option('--learning-rate', type=float, default=1e-4)
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

    train_desc = '{}-lr{:.0e}-bs{:03d}'.format(
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
        monitor='loss')
	

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

    model = DPN((512,512,3))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=learning_rate, momentum=0.9),
                  metrics=['accuracy'])

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
            batch_size=batch_size,
            img_target_size=(512, 512),
            mask_target_size=(512, 512)),
        steps_per_epoch=4,
        epochs=20,
        validation_data=datagen_val.flow_from_list(
            val_img_fnames,
            val_mask_fnames,
            batch_size=4,
            img_target_size=(512, 512),
            mask_target_size=(512, 512)),validation_steps=4)


if __name__ == '__main__':
    train()
