import os
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
from time import gmtime, strftime, localtime
import pandas as pd
from random import shuffle, randint
import logging
import absl.logging as absl_log
import json

import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, roc_curve, auc

from utils import load_config, flatten_config_for_logging, find_threshold, predict_binary
from data_loader.data_loader import (DataGenerator,
                                     get_patches, load_patches)
from data_loader.data_loader import generate_case_partition_manuscript as generate_case_partition
from models.net_keras import *
from data_description.visualization import plot_roc, show_train_history

plt.switch_backend('agg')

# Parse Args
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", default='./configs/base.yml',
                    type=str, help="train configuration")
parser.add_argument("-r", "--run_name", default=None, type=str,
                    help="run name for this experiment. (Default: time)")
args = parser.parse_args()

# Load config
config = load_config(args.config)
if args.run_name is None:
    config['run_name'] = strftime("%Y%m%d_%H%M%S", localtime())
else:
    config['run_name'] = args.run_name
pprint(config)

# Set logging
logging_filename = os.path.join(
    config['log']['checkpoint_dir'], config['run_name'] + '_file.log')
logging.root.removeHandler(absl_log._absl_handler)
absl_log._warn_preinit_stderr = False
logging.basicConfig(
    filename=logging_filename,
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
    format='%(levelname)-8s: %(asctime)-12s: %(message)s'
)

# Set path
model_path = os.path.join(config['log']['model_dir'], config['run_name'])
if not os.path.isdir(model_path):
    os.mkdir(model_path)
result_basepath = os.path.join(
    config['log']['result_dir'], config['run_name'])
if not os.path.isdir(result_basepath):
    os.mkdir(result_basepath)

# Env settings
os.environ["CUDA_VISIBLE_DEVICES"] = config['system']['CUDA_VISIBLE_DEVICES']
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
set_session(tf.compat.v1.Session(config=sess_config))

# split cases into train, val, test
data_partition = generate_case_partition(config)
logging.info("Finish data partition")


# Get train patches
train_X, train_y, train_idx = get_patches(
    config, data_partition, mode='train')

train_X = np.array(train_X)
train_X = train_X.reshape(
    train_X.shape[0], train_X.shape[1], train_X.shape[2], 1)
train_y = np.array(train_y)
print("Finish loading {} patches from {} studies".format(
    train_X.shape[0], len(train_idx)))
logging.info("Finish loading {} patches from {} studies".format(
    train_X.shape[0], len(train_idx)))
print("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(train_y), train_X.shape[0] - np.sum(train_y)))
logging.info("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(train_y), train_X.shape[0] - np.sum(train_y)))

# Get valid patches
valid_X, valid_y, valid_idx = get_patches(
    config, data_partition, mode='validation')

valid_X = np.array(valid_X)
valid_X = valid_X.reshape(
    valid_X.shape[0], valid_X.shape[1], valid_X.shape[2], 1)
valid_y = np.array(valid_y)
print("Finish loading {} patches from {} studies".format(
    valid_X.shape[0], len(valid_idx)))
logging.info("Finish loading {} patches from {} studies".format(
    valid_X.shape[0], len(valid_idx)))
print("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(valid_y), valid_X.shape[0] - np.sum(valid_y)))
logging.info("With {} lesion patches and {} normal pancreas patches".format(
    np.sum(valid_y), valid_X.shape[0] - np.sum(valid_y)))


def AHE(img):
    img[np.where(img != 0)] = img[np.where(img != 0)] + randint(-30, 30)
    return img


# Data Generators - Keras
datagen = ImageDataGenerator(
    horizontal_flip=True,
    fill_mode='constant',
    cval=0.0,
    vertical_flip=True)
# datagen = ImageDataGenerator(
#     preprocessing_function=AHE,
#     horizontal_flip=True,
#     fill_mode='constant',
#     cval=0.0,
#     vertical_flip=True)
datagen.fit(train_X)

# Model Init
model = eval(config['model']['name'])(config['dataset']['input_dim'])
model.compile(
    loss=keras.losses.binary_crossentropy,
    optimizer=keras.optimizers.Adam(lr=config['optimizer']['lr'],
                                    amsgrad=True),
    metrics=['accuracy'])
cbs = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, verbose=1),
    keras.callbacks.EarlyStopping(monitor='val_loss', patience=40)
]

# Model Training
class_weights = class_weight.compute_class_weight(
    'balanced', np.unique(train_y), train_y)
print("Setting class weights {}".format(class_weights))
logging.info("Setting class weights {}".format(class_weights))

logging.info("Start training")
history = model.fit_generator(
    datagen.flow(train_X, train_y, batch_size=config['train']['batch_size']),
    epochs=config['train']['epochs'],
    callbacks=cbs,
    steps_per_epoch=len(train_X) / config['train']['batch_size'],
    class_weight=class_weights,
    validation_data=(valid_X, valid_y))
logging.info("Finish training")

model.save_weights(os.path.join(model_path, 'weights.h5'))
logging.info("Finish saving model")

# Find patch-based threshold from validation data
valid_probs = model.predict_proba(valid_X)
patch_threshold = find_threshold(valid_probs, valid_y)
print("Patch threshold: {}".format(patch_threshold))
logging.info("Patch threshold: {}".format(patch_threshold))

# Fine patient-based threshold from training and validation data
train_probs = model.predict_proba(train_X)
train_pred = predict_binary(train_probs, patch_threshold)
valid_pred = predict_binary(valid_probs, patch_threshold)
patient_y = []
patient_probs = []
pre_idx = 0
cur_idx = 0
for patient in train_idx:
    cur_idx = cur_idx + patient[2]
    pred_y = train_pred[pre_idx:cur_idx]
    true_y = train_y[pre_idx:cur_idx]
    pre_idx = cur_idx
    matrix = confusion_matrix(true_y, pred_y, labels=[1, 0])
    if patient[0] == 'msd' or patient[1][:2] == 'PC' or patient[1][:2] == 'PT':
        patient_y.append(1)
        patient_probs.append(matrix[0][0] / np.sum(true_y))
    else:
        patient_y.append(0)
        patient_probs.append(matrix[1][0] / (matrix[1][0] + matrix[1][1]))
pre_idx = 0
cur_idx = 0
for patient in valid_idx:
    cur_idx = cur_idx + patient[2]
    pred_y = valid_pred[pre_idx:cur_idx]
    true_y = valid_y[pre_idx:cur_idx]
    pre_idx = cur_idx
    matrix = confusion_matrix(true_y, pred_y, labels=[1, 0])
    if patient[0] == 'msd' or patient[1][:2] == 'PC' or patient[1][:2] == 'PT':
        patient_y.append(1)
        patient_probs.append(matrix[0][0] / np.sum(true_y))
    else:
        patient_y.append(0)
        patient_probs.append(matrix[1][0] / (matrix[1][0] + matrix[1][1]))
patient_threshold = find_threshold(patient_probs, patient_y)
print("Patient threshold: {}".format(patient_threshold))
logging.info("Patient threshold: {}".format(patient_threshold))
info = {}
info['partition'] = data_partition
info['patch_threshold'] = float(patch_threshold)
info['patient_threshold'] = float(patient_threshold)
info['config'] = config
with open(os.path.join(config['log']['checkpoint_dir'], (config['run_name'] + '_info.json')), 'w') as f:
    json.dump(info, f)

# Save the result
fig_acc = show_train_history(history, 'acc', 'val_acc')
plt.savefig(os.path.join(result_basepath, 'acc_plot.png'))

fig_los = show_train_history(history, 'loss', 'val_loss')
plt.savefig(os.path.join(result_basepath, 'loss_plot.png'))
