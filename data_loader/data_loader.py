import glob
import os

import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import nibabel as nib
from random import shuffle
import SimpleITK as sitk


from data_loader.patch_sampler import patch_generator
from data_loader.patch_sampler import masked_2D_sampler
from data_loader.preprocessing import (
    minmax_normalization, windowing, smoothing, pancreas_normalization)
from data_loader.create_boxdata import finecut_to_thickcut


class DataGenerator():
    '''
    Data generator for external dataset
    Args:
        patch_size (int): patch size
        stride (int): distance of moving window
        data_type (str): 'NTUH' or 'MSD' or 'Pancreas-CT'
    '''

    def __init__(self, patch_size=50, stride=5, data_type='MSD'):
        self.patch_size = patch_size
        self.stride = stride
        self.data_type = data_type
        if data_type == 'tcia':
            self.data_path = '/data/pancreas/Pancreas-CT/'
        elif data_type == 'msd':
            self.data_path = '/data/pancreas/MSD/Task07_Pancreas/'
        elif data_type == 'ntuh':
            self.data_path = '/data2/pancreas/Nifti_data/'

    def load_image(self, filename):
        self.filename = filename
        if self.data_type == 'tcia':
            file_location = glob.glob(os.path.join(
                self.data_path, filename) + '/*/*/000001.dcm')
            imagedir = os.path.dirname(file_location[0])
            labelname = 'label' + filename[-4:] + '.nii.gz'
            labelpath = os.path.join(self.data_path, 'annotations', labelname)

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(imagedir)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()
            image_array = sitk.GetArrayFromImage(image).transpose((2, 1, 0))
            self.thickness = image.GetSpacing()[2]
            # TODO: add affine
            label = nib.load(labelpath).get_data()

        elif self.data_type == 'msd':
            imagepath = os.path.join(self.data_path, 'imagesTr', filename)
            labelpath = os.path.join(self.data_path, 'labelsTr', filename)

            img = nib.load(imagepath)
            image_array = img.get_data()
            label = nib.load(labelpath).get_data()
            self.thickness = img.affine[2, 2]
            self.affine = img.affine

        elif self.data_type == 'ntuh':
            imagepath = os.path.join(
                self.data_path, 'image', 'IM_' + filename + '.nii.gz')
            labelpath = os.path.join(
                self.data_path, 'label', 'LB_' + filename + '.nii.gz')
            backup_path = '/data2/pancreas/results/image_AD/'
            rothpath = os.path.join(backup_path, 'IM_'
                                    + filename + '/IM_' + filename + '_model.nii.gz')

            img = nib.load(imagepath)
            image_array = img.get_data()
            if os.path.exists(labelpath):
                label = nib.load(labelpath).get_data()
            elif os.path.exists(rothpath):
                result = nib.load(rothpath).get_data()
                label = np.zeros(result.shape)
                label[np.where(result == 8)] = 1
            else:
                print("can't find label file!{}".format(rothpath))
            self.thickness = img.affine[2, 2]
            self.affine = img.affine

        return image_array, label

    def load_box(self, filename, box_path):
        self.filename = filename
        if self.data_type == 'tcia':
            file_dir = os.path.join(box_path, 'Pancreas-CT')
            file_path = os.path.join(file_dir, filename)
        elif self.data_type == 'msd':
            file_dir = os.path.join(box_path, 'MSD')
            file_path = os.path.join(file_dir, filename)
        elif self.data_type == 'ntuh':
            file_dir = os.path.join(box_path, 'NTUH')
            file_path = os.path.join(file_dir, filename)
        image = np.load(os.path.join(file_path, 'ctscan.npy'))
        pancreas = np.load(os.path.join(file_path, 'pancreas.npy'))
        lesion = np.load(os.path.join(file_path, 'lesion.npy'))
        return image, pancreas, lesion

    def get_boxdata(self, image, label, border=(10, 10, 3)):

        # Transfer label
        pancreas = np.zeros(label.shape)
        pancreas[np.where(label != 0)] = 1
        lesion = np.zeros(label.shape)
        if self.data_type == 'msd' or self.data_type == 'ntuh':
            lesion[np.where(label == 2)] = 1

        # Finecut to thickcut
        if self.thickness < 5:
            image = finecut_to_thickcut(image, self.thickness)
            pancreas = finecut_to_thickcut(
                pancreas, self.thickness, label_mode=True)
            lesion = finecut_to_thickcut(
                lesion, self.thickness, label_mode=True)

        # Generate box index
        xmin, ymin, zmin = np.max(
            [np.min(np.where(pancreas != 0), 1) - border, (0, 0, 0)], 0)
        xmax, ymax, zmax = np.min(
            [np.max(np.where(pancreas != 0), 1) + border, label.shape], 0)

        # Generate box data
        box_image = image[xmin:xmax, ymin:ymax, zmin:zmax]
        box_pancreas = pancreas[xmin:xmax, ymin:ymax, zmin:zmax]
        box_lesion = lesion[xmin:xmax, ymin:ymax, zmin:zmax]

        return box_image, box_pancreas, box_lesion

    def preprocessing(self, image, pancreas, lesion):
        from skimage import morphology

        if self.data_type == 'tcia':
            image = image[:, ::-1, :]
            pancreas = pancreas[:, ::-1, :]
            lesion = lesion[:, ::-1, :]
            pancreas = smoothing(pancreas)
            lesion = smoothing(lesion)
        elif self.data_type == 'msd':
            image = image[::-1, ::-1, :]
            pancreas = pancreas[::-1, ::-1, :]
            lesion = lesion[::-1, ::-1, :]
        elif self.data_type == 'ntuh':
            image = image[::-1, ::-1, :]
            pancreas = pancreas[::-1, ::-1, :]
            lesion = lesion[::-1, ::-1, :]

        pancreas = morphology.dilation(pancreas, np.ones([3, 3, 1]))
        lesion = morphology.dilation(lesion, np.ones([3, 3, 1]))

        image = windowing(image)
        image = minmax_normalization(image)

        return image, pancreas, lesion

    def preprocessing_test(self, image, pancreas, lesion):
        from skimage import morphology

        if self.data_type == 'tcia':
            image = image[:, ::-1, :]
            pancreas = pancreas[:, ::-1, :]
            lesion = lesion[:, ::-1, :]
            pancreas = smoothing(pancreas)
            lesion = smoothing(lesion)
        elif self.data_type == 'msd':
            image = image[::-1, ::-1, :]
            pancreas = pancreas[::-1, ::-1, :]
            lesion = lesion[::-1, ::-1, :]
        elif self.data_type == 'ntuh':
            image = image[::-1, ::-1, :]
            pancreas = pancreas[::-1, ::-1, :]
            lesion = lesion[::-1, ::-1, :]

        pancreas = morphology.dilation(pancreas, np.ones([3, 3, 1]))
        lesion = morphology.dilation(lesion, np.ones([3, 3, 1]))

        image = windowing(image)
        image = pancreas_normalization(image, pancreas, lesion)

        return image, pancreas, lesion

    def generate_patch(self, image, pancreas, lesion, y_type='bin', mask=False):
        X = []
        Y = []

        self.coords = masked_2D_sampler(lesion, pancreas,
                                        self.patch_size, self.stride,
                                        threshold=1 / (self.patch_size ** 2),
                                        y_type=y_type)

        if mask:
            image[np.where(pancreas == 0)] = 0

        for coord in self.coords:
            mask_pancreas = image[coord[1]
                                  :coord[4], coord[2]:coord[5], coord[3]]
            X.append(mask_pancreas)
            Y.append(coord[0])

        self.X = X
        self.Y = Y

        return X, Y

    def patch_len(self):
        return len(self.X)

    def gt_pancreas_num(self):
        return len(self.Y) - np.sum(self.Y)

    def gt_lesion_num(self):
        return np.sum(self.Y)

    def get_prediction(self, model, patch_threshold=0.5):
        from sklearn.metrics import confusion_matrix
        if len(self.X) > 0:
            test_X = np.array(self.X)
            test_X = test_X.reshape(
                (test_X.shape[0], test_X.shape[1], test_X.shape[2], 1))
            test_Y = np.array(self.Y)
        else:
            return filename

        self.probs = model.predict_proba(test_X)
        predict_y = predict_binary(self.probs, patch_threshold)
        self.patch_matrix = confusion_matrix(test_Y, predict_y, labels=[1, 0])

        return self.patch_matrix

    def get_auc(self):
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, thresholds = roc_curve(self.Y, self.probs)
        return auc(fpr, tpr)

    def get_probs(self):
        return self.probs, self.Y

    def get_roc_curve(self):
        from data_description.visualization import plot_roc
        plot_roc(self.probs, self.Y)

    def get_all_value(self):
        tp = self.patch_matrix[0][0]
        fn = self.patch_matrix[0][1]
        fp = self.patch_matrix[1][0]
        tn = self.patch_matrix[1][1]
        return tp, fn, fp, tn


def get_patches(config, case_partition, mode='train', y_type='bin'):
    X = []
    y = []
    idx = []
    patch_size = config['dataset']['input_dim'][0]
    stride = config['dataset']['stride']
    load_way = config['dataset']['load']
    mask = config['dataset']['mask']

    for partition in case_partition:
        print("GET_PATCHES:\tLoading {} data ... ".format(partition['type']))
        datagen = DataGenerator(
            patch_size, stride=stride, data_type=partition['type'])
        if load_way == 'ori':
            print("GET_PATCHES:\tLoading image from origin dataset")
            for case_id in tqdm(partition[mode]):
                img, lbl = datagen.load_image(case_id)
                box_img, box_pan, box_les = datagen.get_boxdata(
                    img, lbl)
                image, pancreas, lesion = datagen.preprocessing(
                    box_img, box_pan, box_les)
                tmp_X, tmp_y = datagen.generate_patch(
                    image, pancreas, lesion, y_type=y_type, mask=mask)
                X = X + tmp_X
                y = y + tmp_y
        elif load_way == 'ori_box':
            print("GET_PATCHES:\tLoading box data from {}".format(
                config['dataset']['box_dir']))
            for case_id in tqdm(partition[mode]):
                box_img, box_pan, box_les = datagen.load_box(
                    case_id, box_path=config['dataset']['box_dir'])
                image, pancreas, lesion = datagen.preprocessing(
                    box_img, box_pan, box_les)
                tmp_X, tmp_y = datagen.generate_patch(
                    image, pancreas, lesion, y_type=y_type, mask=mask)
                X = X + tmp_X
                y = y + tmp_y
                idx.append([partition['type'], case_id, len(tmp_y)])
        elif load_way == 'box':
            print("GET_PATCHES:\tLoading preprocessed box data from {}".format(
                config['dataset']['box_dir']))
            for case_id in tqdm(partition[mode]):
                image, pancreas, lesion = datagen.load_box(
                    case_id, box_path=config['dataset']['box_dir'])
                tmp_X, tmp_y = datagen.generate_patch(
                    image, pancreas, lesion, y_type=y_type, mask=mask)
                X = X + tmp_X
                y = y + tmp_y
                idx.append([partition['type'], case_id, len(tmp_y)])

    return X, y, idx


def predict_binary(prob, threshold):
    binary = np.zeros(prob.shape)
    binary[prob < threshold] = 0
    binary[prob >= threshold] = 1
    return binary


def split_save_case_partition(case_list, ratio=(0.8, 0.1, 0.1), path=None, test_cases=None, random_seed=None):
    """Splting all cases to train, val, test part

    If path is not empty str, partition dict is saved for reproducibility.

    Args:
        case_list (list): The list contains case name.
        ratio (tup): Data split ratio. SHOULD sum to 1. (train, val, test) Defaults to (.8, .1, .1).
        path (str): Path to Save the partition dict for reproducibility.
        test_cases (list): For fixing the testing cases.
        random_seed (int): Random Seed.


    Returns:
        dict:   Keys: {all, train, validation, test}
                Values: list of cases

    """

    print('SPLIT_SAVE_CASE_PARTITION:\tStart spliting cases...')

    partition = {}
    if test_cases is None:
        partition['all'] = case_list
        # load case list and spilt to 3 part
        print('SPLIT_SAVE_CASE_PARTITION:\tTarget Partition Ratio: (train, val, test)={}'.format(ratio))
        partition['train'], partition['test'] = train_test_split(
            partition['all'], test_size=ratio[2], random_state=random_seed)
        partition['train'], partition['validation'] = train_test_split(
            partition['train'], test_size=ratio[1] / (ratio[0] + ratio[1]), random_state=random_seed)
    elif type(test_cases) is list:
        # load predifined test cases
        partition['all'] = list(set(case_list + test_cases))
        print('SPLIT_SAVE_CASE_PARTITION:\tUsing PREDEFINED TEST CASES')
        partition['test'] = test_cases
        train_case = list(set(case_list) - set(test_cases))
        partition['train'], partition['validation'] = train_test_split(
            train_case, test_size=ratio[1] / (ratio[0] + ratio[1]), random_state=random_seed)
    else:
        raise TypeError(
            "test_cases expected to be \"list\", instead got {}".format(type(test_cases)))

    # report actual partition ratio
    num_parts = list(map(len, [partition[part]
                               for part in ['train', 'validation', 'test']]))
    print('SPLIT_SAVE_CASE_PARTITION:\tActual Partition Number: (train, val, test)={}'.format(
        (num_parts)))

    print('SPLIT_SAVE_CASE_PARTITION:\tDone Partition')
    # saving partition dict to disk
    if path is not None:
        print('SPLIT_SAVE_CASE_PARTITION:\tStart saving partition dict to {}'.format(path))
        with open(path, 'wb') as f:
            pickle.dump(partition, f, pickle.HIGHEST_PROTOCOL)
        print('SPLIT_SAVE_CASE_PARTITION:\tDone saving')

    return partition


def split_case_partition(case_list, value=(10, 10, 10), path=None, test_cases=None, random_seed=None):
    """Splting all cases to train, val, test part

    If path is not empty str, partition dict is saved for reproducibility.

    Args:
        case_list (list): The list contains case name.
        ratio (tup): Data split ratio. SHOULD sum to 1. (train, val, test) Defaults to (.8, .1, .1).
        path (str): Path to Save the partition dict for reproducibility.
        test_cases (list): For fixing the testing cases.
        random_seed (int): Random Seed.


    Returns:
        dict:   Keys: {all, train, validation, test}
                Values: list of cases

    """
    import random
    print('SPLIT_CASE_PARTITION:\tStart spliting cases...')

    assert (value[0] + value[1] + value[2]
            ) < len(case_list) + 1, 'SPLIT_CASE_PARTITION:\ttotal length exceed!'

    partition = {}
    random.Random(random_seed).shuffle(case_list)
    if test_cases is None:
        partition['all'] = case_list[: value[0] + value[1] + value[2]]
        partition['test'] = case_list[: value[2]]
        partition['validation'] = case_list[value[2] : value[1] + value[2]]
        partition['train'] = case_list[value[1]
                                       + value[2] : value[0] + value[1] + value[2]]
    else:
        raise TypeError(
            "test_cases expected to be \"list\", instead got {}".format(type(test_cases)))

    # report actual partition ratio
    num_parts = list(map(len, [partition[part]
                               for part in ['train', 'validation', 'test']]))
    print('SPLIT_CASE_PARTITION:\tActual Partition Number: (train, val, test)={}'.format(
        (num_parts)))

    print('SPLIT_CASE_PARTITION:\tDone Partition')
    # saving partition dict to disk
    if path is not None:
        print('SPLIT_SAVE_CASE_PARTITION:\tStart saving partition dict to {}'.format(path))
        with open(path, 'wb') as f:
            pickle.dump(partition, f, pickle.HIGHEST_PROTOCOL)
        print('SPLIT_SAVE_CASE_PARTITION:\tDone saving')

    return partition


def split_fold_partition(case_list, test_value, fold, path=None, test_cases=None, random_seed=None):
    """Splting all cases to each fold and test part

    If path is not empty str, partition dict is saved for reproducibility.

    Args:
        case_list (list): The list contains case name.
        test_value (int): Test data amount.
        path (str): Path to Save the partition dict for reproducibility.
        test_cases (list): For fixing the testing cases.
        random_seed (int): Random Seed.


    Returns:
        dict:   Keys: {all, train, validation, test}
                Values: list of cases

    """
    import random
    print('SPLIT_CASE_PARTITION:\tStart spliting cases...')

    split_number = (len(case_list) - test_value) // fold

    print(len(case_list))
    random.Random(random_seed).shuffle(case_list)
    data_partition = []
    for idx in range(fold):
        partition = {}
        partition['all'] = case_list
        partition['test'] = case_list[: test_value]
        if idx + 1 == fold:
            partition['validation'] = case_list[test_value + idx * split_number :]
        else:
            partition['validation'] = case_list[test_value + idx * split_number : test_value + (idx + 1) * split_number]
        partition['train'] = case_list.copy()
        for case_id in partition['test']:
            partition['train'].remove(case_id)
        for case_id in partition['validation']:
            partition['train'].remove(case_id)
        partition['fold'] = idx + 1
        data_partition.append(partition)

        # report actual partition ratio
        num_parts = list(map(len, [partition[part]
                                for part in ['train', 'validation', 'test']]))
        print('SPLIT_CASE_PARTITION:\tActual Partition Number: (train, val, test)={}'.format(
            (num_parts)))

    print('SPLIT_CASE_PARTITION:\tDone Partition')
    # saving partition dict to disk
    if path is not None:
        print('SPLIT_SAVE_CASE_PARTITION:\tStart saving partition dict to {}'.format(path))
        with open(path, 'wb') as f:
            pickle.dump(partition, f, pickle.HIGHEST_PROTOCOL)
        print('SPLIT_SAVE_CASE_PARTITION:\tDone saving')

    return data_partition


def load_case_partition(path):
    """Load the cases partition

    Args:
        path (str): Path to partition dict.


    Returns:
        dict:   Keys: {all, train, validation, test}
                Values: list of cases

    """

    print('LOAD_CASE_PARTITION:\tStart loading case partition...')

    # loading partition dict from disk
    with open(path, 'rb') as f:
        partition = pickle.load(f)

    # report partiton ratio
    num_parts = list(map(len, [partition[part]
                               for part in ['train', 'validation', 'test']]))
    ratio = num_parts / sum(num_parts)
    print('LOAD_CASE_PARTITION:\tPartition Ratio: (train, val, test)={}'.format((ratio)))

    print('LOAD_CASE_PARTITION:\tDone loading case partition')
    return partition


def get_patch_partition_labels(case_partition, pancreas_dir, lesion_dir):
    """Splting patches based on case partition.

    patch_id = case_pathIndex


    Returns:
        dict:   Keys: {all, train, validation, test}
                Values: list of patch id
        dict:   Keys: patch_id
                Values: abs. path of patch
        dict:   Keys: patch_id
                Values: label

    """

    patch_partition = {'train': [], 'validation': [], 'test': []}
    patch_paths = {'train': [], 'validation': [], 'test': []}
    labels = {}

    print('GET_PATCH_PARTITION_LABELS:\tStart loading patch partition...')
    for part in ['train', 'validation', 'test']:
        print('GET_PATCH_PARTITION_LABELS:\t Progress: {}'.format(part))
        for case in case_partition[part]:
            for i, path_dir in enumerate([pancreas_dir, lesion_dir]):
                for patch_path in glob.glob(path_dir + '/' + case + '_*.npy'):
                    patch_id = patch_path.split('/')[-1].split('.')[0]
                    patch_partition[part].append(patch_id)
                    patch_paths[patch_id] = patch_path
                    labels[patch_id] = i
    print('GET_PATCH_PARTITION_LABELS:\tDone patch partition')
    return patch_partition, patch_paths, labels


def load_patches(data_path, case_list, patch_size=50, stride=5):
    X_total = []
    y_total = []
    for ID in tqdm.tqdm(case_list):
        X_tmp, y_tmp = patch_generator(
            data_path, ID, patch_size, stride=stride, threshold=0.0004, max_amount=1000)
        X_total.extend(X_tmp)
        y_total.extend(y_tmp)
    X = np.array(X_total)
    X = X.reshape((X.shape[0], X.shape[1], X.shape[2], 1))
    y = np.array(y_total)

    return X, y


def convert_csv_to_dict(csv_data_path='/data2/pancreas/raw_data/data_list.csv'):
    """
    version: 2019/03
    extract data list of train, validation, test from csv
    csv_data_path = path to the csv file (ex: '/home/d/pancreas/raw_data/data_list.csv')
    Returns:
        dict:   Keys: {all, train, validation, test}
                Values: list of patch id
    """
    final_split_df = pd.read_csv(csv_data_path)
    data_list_dict = {}
    data_list_dict['train'] = list(
        final_split_df[final_split_df['Class'] == 'train']['Number'])
    data_list_dict['validation'] = list(
        final_split_df[final_split_df['Class'] == 'validation']['Number'])
    data_list_dict['test'] = list(
        final_split_df[final_split_df['Class'] == 'test']['Number'])
    data_list_dict['all'] = list(final_split_df[final_split_df['Class'] == 'train']['Number']) + \
        list(final_split_df[final_split_df['Class'] == 'validation']['Number']) + \
        list(final_split_df[final_split_df['Class'] == 'test']['Number'])
    print('Finish converting csv to dict')
    return data_list_dict


def load_list(list_path):
    '''
    version: 2019/08
    extract data list of train and test from csv
    training data: 100 healthy and 100 tumor
    testing data: 80 healthy and 80 tumor
    '''
    df = pd.read_csv(list_path, converters={'add_date': str})
    data_list_dict = {}

    healthy_total = df[(df['type'] == 'healthy')
                       & (df['diff_patient_list'] == True)]
    healthy_train = list(
        healthy_total[healthy_total['add_date'] == '20190210']['case_id'])
    healthy_train.remove('AD54')
    healthy_train.remove('AD95')
    healthy_test = list(
        healthy_total[healthy_total['add_date'] == '20190618']['case_id'])
    healthy_test.remove('AD137')
    healthy_test.remove('AD143')

    tumor_total = df[(df['type'] == 'tumor') & (
        df['diff_patient_list'] == True)]
    tumor_train = list(
        tumor_total[tumor_total['add_date'] == '20190210']['case_id'])
    tumor_train.remove('PC47')
    tumor_test = list(
        tumor_total[tumor_total['add_date'] == '20190618']['case_id'])
    tumor_test.remove('PC570')
    tumor_test.remove('PC653')

    shuffle(healthy_train)
    shuffle(tumor_train)

    data_list_dict['healthy_train'] = healthy_train
    data_list_dict['healthy_test'] = healthy_test
    data_list_dict['tumor_train'] = tumor_train
    data_list_dict['tumor_test'] = tumor_test

    return data_list_dict


def generate_case_partition(config):
    '''
    Version: 2019/11
    extract NTUH data list from csv
    extract MSD, Pancreas-CT data list
    '''

    list_path = config['dataset']['csv']
    df = pd.read_csv(list_path, converters={'add_date': str})

    healthy_list = list(df[(df['type'] == 'healthy')
                           & (df['diff_patient_list'])]['case_id'])

    tumor_list = list(df[(df['type'] == 'tumor')
                         & (df['diff_patient_list'])]['case_id'])

    healthy_partition = split_case_partition(
        healthy_list, value=config['dataset']['NTUH_split'], random_seed=config['dataset']['seed'])

    tumor_partition = split_case_partition(
        tumor_list, value=config['dataset']['NTUH_split'], random_seed=config['dataset']['seed'])

    ntuh_partition = {}
    ntuh_partition['type'] = 'ntuh'
    ntuh_partition['all'] = healthy_list + tumor_list
    ntuh_partition['validation'] = healthy_partition['validation'] + \
        tumor_partition['validation']
    ntuh_partition['train'] = healthy_partition['train'] + \
        tumor_partition['train']
    ntuh_partition['test'] = healthy_partition['test'] + tumor_partition['test']

    data_path_pancreasct = '/data/pancreas/Pancreas-CT/'
    data_path_msd = '/data/pancreas/MSD/Task07_Pancreas/'

    tcia_list = os.listdir(data_path_pancreasct)
    for filename in tcia_list:
        if not filename[0] == 'P':
            tcia_list.remove(filename)
    tcia_partition = split_case_partition(
        tcia_list, value=config['dataset']['PancreasCT_split'], random_seed=config['dataset']['seed'])
    tcia_partition['type'] = 'tcia'
    tcia_partition['all'] = tcia_list

    msd_list = os.listdir(os.path.join(data_path_msd, 'imagesTr'))
    for filename in msd_list:
        if not filename[0] == 'p':
            msd_list.remove(filename)
    msd_partition = split_case_partition(
        msd_list, value=config['dataset']['MSD_split'], random_seed=config['dataset']['seed'])
    msd_partition['type'] = 'msd'
    msd_partition['all'] = msd_list

    return [ntuh_partition, tcia_partition, msd_partition]


def generate_case_partition_manuscript(config):
    '''
    Version: 2019/12
    extract NTUH data list from csv
    extract MSD, Pancreas-CT data list
    '''

    list_path = config['dataset']['csv']
    df = pd.read_csv(list_path, converters={'add_date': str})

    healthy_list = list(df[(df['type'] == 'healthy')
                           & (df['diff_patient_list'])]['case_id'])

    tumor_list = list(df[(df['type'] == 'tumor')
                         & (df['diff_patient_list'])
                         & (df['exam_date'] > 20130000)]['case_id'])

    healthy_partition = split_case_partition(
        healthy_list, value=config['dataset']['split']['NTUH_healthy'], random_seed=config['dataset']['seed'])

    tumor_partition = split_case_partition(
        tumor_list, value=config['dataset']['split']['NTUH_tumor'], random_seed=config['dataset']['seed'])

    ntuh_partition = {}
    ntuh_partition['type'] = 'ntuh'
    ntuh_partition['all'] = healthy_list + tumor_list
    ntuh_partition['validation'] = healthy_partition['validation'] + \
        tumor_partition['validation']
    ntuh_partition['train'] = healthy_partition['train'] + \
        tumor_partition['train']
    ntuh_partition['test'] = healthy_partition['test'] + tumor_partition['test']

    return [ntuh_partition]


def generate_cv_partition_manuscript(config, fold=5):
    '''
    Version: 2019/12
    extract NTUH data list
    generate cross validation partition
    '''

    list_path = config['dataset']['csv']
    df = pd.read_csv(list_path, converters={'add_date': str})

    healthy_list = list(df[(df['type'] == 'healthy')
                           & (df['diff_patient_list'])]['case_id'])[:320]

    tumor_list = list(df[(df['type'] == 'tumor')
                         & (df['diff_patient_list'])
                         & (df['exam_date'] > 20130000)]['case_id'])

    healthy_partition = split_fold_partition(
        healthy_list, test_value=config['dataset']['split']['NTUH_healthy'][2],
        fold=fold, random_seed=config['dataset']['seed'])

    tumor_partition = split_fold_partition(
        tumor_list, test_value=config['dataset']['split']['NTUH_tumor'][2],
        fold=fold, random_seed=config['dataset']['seed'])

    case_partition = []
    for idx in range(fold):
        partition = {}
        partition['type'] = 'ntuh'
        partition['all'] = healthy_partition[idx]['all'] + \
            tumor_partition[idx]['all']
        partition['train'] = healthy_partition[idx]['train'] + \
            tumor_partition[idx]['train']
        partition['validation'] = healthy_partition[idx]['validation'] + \
            tumor_partition[idx]['validation']
        partition['test'] = healthy_partition[idx]['test'] + \
            tumor_partition[idx]['test']
        partition['fold'] = idx + 1
        case_partition.append(partition)

    return case_partition
