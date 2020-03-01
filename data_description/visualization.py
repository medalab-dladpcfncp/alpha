from sklearn import metrics
import pydicom as dicom
import pandas as pd
import numpy as np
import os
import glob

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def draw_roc(gt, pd, save_name=None, exp_name=None):
    fpr, tpr, threshold = roc_curve(gt, pd)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right', prop={'size': 14})

    plt.ylabel('Sensitivity', fontsize=14)
    plt.xlabel('1 - Specificity', fontsize=14)
    if save_name:
        plt.savefig(os.path.join('../result', exp_name, save_name + '.png'), bbox_inches='tight')


def plot_roc(probs, test_y):
    """
    Usage: Plot roc curves

    Parameters
    ----------
    probs:
    test_y:

    Returns
    -------
    figure
    """
    fpr, tpr, threshold = metrics.roc_curve(test_y, probs)
    roc_auc = metrics.auc(fpr, tpr)

    plt.figure()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def show_train_history(train_history, train, validation):
    plt.figure()
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
