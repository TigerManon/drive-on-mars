import os

import numpy as np
import pandas as pd

import cv2

# from params import IMAGE_PATH, MASK_ROVER, RANGE_30M, MASK_PATH_TRAIN


# Customize your path here
root_path = '../'

DATA_PATH = os.path.join(root_path, 'raw_data', 'ai4mars-dataset-merged-0.1', 'msl')

IMAGE_PATH = os.path.join(DATA_PATH, 'images', 'edr')
MASK_PATH_TRAIN = os.path.join(DATA_PATH, 'labels', 'train')
MASK_PATH_TESTS = os.path.join(DATA_PATH, 'labels', 'test')
TESTS_DIR = ['masked-gold-min1-100agree', 'masked-gold-min2-100agree', 'masked-gold-min3-100agree']

MASK_ROVER = os.path.join(DATA_PATH, 'images', 'mxy')
RANGE_30M = os.path.join(DATA_PATH, 'images', 'rng-30m')



def create_df(path):
    """
    Generate dataframe based on all labeled images in a directory
    such as train labels or test labels.
    Input: path
    Output: dataframe with columns
        name of raw image
        name of the labeled image
        name of rover mask image
        name of range 30m mask image
    """
    names = []
    labels = []

    for filename in sorted(os.listdir(path)):
        imname = filename.split('.')[0]

        label = imname
        imname = imname.replace("_merged","")

        # checking raw image
        raw_image_file = os.path.join(IMAGE_PATH,imname + ".JPG")
        if not os.path.isfile(raw_image_file):
            print('No raw image found for', imname)
            continue

        # checking rover masks
        mask_rover_file = os.path.join(MASK_ROVER,imname.replace("EDR","MXY") + ".png")
        if not os.path.isfile(mask_rover_file):
            print('No rover mask found for', imname)
            continue

        # checking range masks
        range_mask_file = os.path.join(RANGE_30M, imname.replace("EDR","RNG") + ".png")
        if not os.path.isfile(range_mask_file):
            print('No range mask found for', imname)
            continue

        names.append(imname)
        labels.append(label)

    df = pd.DataFrame(
        {'name': names, 'label': labels},
        index = np.arange(0, len(names)))
    df['rov_mask'] = df.name.apply(lambda imname: imname.replace("EDR","MXY"))
    df['rang_mask'] = df.name.apply(lambda imname: imname.replace("EDR","RNG"))

    return df


def load_image_set(im_id, df):
    """
    Returns list of 2D arrays for each image as part of a set
    Input:  index of the image dataset dataframe
            the dataframe for the image dataset (train or test)
    Output:
            raw image
            labeled image

    """

    # Load raw image
    edr_file = f'{df.name.iloc[im_id]}.JPG'
    image_raw = cv2.imread(os.path.join(IMAGE_PATH,edr_file))
    image = image_raw[:,:,0]

    # Load labels
    label_file = f'{df.label.iloc[im_id]}.png'
    label_raw = cv2.imread(os.path.join(MASK_PATH_TRAIN,label_file))
    label = label_raw[:,:,0]

    # Changing scale for the 'No label' encoded as 255
    label[label == 255] = 4

    # Load and combine both masks
    rov_mask_file = f'{df.rov_mask.iloc[im_id]}.png'
    rov_mask_raw = np.array(cv2.imread(os.path.join(MASK_ROVER,rov_mask_file)))
    rov_mask = np.zeros((1024,1024))
    rov_mask[:,:] = rov_mask_raw[:,:,0]

    rang_mask_file = f'{df.rang_mask.iloc[im_id]}.png'
    rang_mask_raw = np.array(cv2.imread(os.path.join(RANGE_30M,rang_mask_file)))
    rang_mask = np.zeros((1024,1024))
    rang_mask[:,:] = rang_mask_raw[:,:,0]

    # reversing mask to only keep the image out of the mask
    mask = (1-rov_mask) * (1-rang_mask)

    return [image, label, mask]



def decompose_label(label_img):

    label_0 = np.where(label_img == 0, 1, 0)
    label_1 = np.where(label_img == 1, 1, 0)
    label_2 = np.where(label_img == 2, 1, 0)
    label_3 = np.where(label_img == 3, 1, 0)
    label_4 = np.where(label_img == 4, 1, 0)

    return np.stack([label_0, label_1, label_2, label_3, label_4], axis=-1)


def load_images(df):
    """
    Aggregates images and labels with masks applied
    Input: Dataframe of train or test sets
    Output: Array of
        masked images from raws
        masked labels
    """
    X, y = [], []
    for i in df.index:
        [image, label, mask] = load_image_set(i, df)

        X.append(image * mask)
        y.append(label * mask)

    X = np.array(X)
    y = np.array(y)
    print('✅ loaded raw images and labels')
    y_decomp = decompose_label(y)
    print('✅ decomposed labels into binary masks')

    return X, y_decomp


if __name__ == '__main__':

    small_dataset = 100

    df = create_df(MASK_PATH_TRAIN)

    df = df.iloc[:100]

    X, y = load_images(df)
