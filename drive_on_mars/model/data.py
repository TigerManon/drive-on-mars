import os

import numpy as np
import pandas as pd

import cv2

from drive_on_mars.params import IMAGE_PATH, MASK_ROVER, RANGE_30M, MASK_PATH_TRAIN, RESIZE_SHAPE


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


def preproc_image(image_file):
    """
    Each image is set to a single channel greyscale and resized
    """
    image_raw = cv2.imread(image_file)
    image = image_raw
    image = cv2.resize(image, dsize = (RESIZE_SHAPE, RESIZE_SHAPE))

    return image


def load_mask(im_id, df):
    """
    Loading both range and rover masks combined
    """
    # Load and combine both masks with the proper resize
    rov_mask_file = f'{df.rov_mask.iloc[im_id]}.png'
    rov_mask = preproc_image(os.path.join(MASK_ROVER,rov_mask_file))
    rang_mask_file = f'{df.rang_mask.iloc[im_id]}.png'
    rang_mask = preproc_image(os.path.join(RANGE_30M,rang_mask_file))

    # reversing mask to only keep the image out of the mask
    mask = (1-rov_mask) * (1-rang_mask)

    return mask


def preproc(df, use_mask = False, write_output=False):
    """
    DEV: Method to preprocess the data and save the output
    """
    for im_id in df.index:
        # Load raw image
        img_name = f'{df.name.iloc[im_id]}.JPG'
        image = preproc_image(img_name)
        preproc_img_name = f'{df.name.iloc[im_id]}_preproc.JPG'

        if use_mask:
            mask = load_mask(im_id, df)
            image = image * mask
            preproc_img_name = f'{df.name.iloc[im_id]}_masked.JPG'

        if write_output:
            cv2.imwrite(preproc_img_name, image)



def load_preproc(im_id, df, use_mask = False):
    """
    Returns list of 2D arrays for each image as part of a set
    Input:  index of the image dataset dataframe
            the dataframe for the image dataset (train or test)
    Output:
            image
            labeled image
    """

    # Load raw image
    edr_file = os.path.join(IMAGE_PATH,edr_file, f'{df.name.iloc[im_id]}.JPG')
    image = preproc_image(edr_file)

    # Load labels
    label_file = os.path.join(MASK_PATH_TRAIN, f'{df.label.iloc[im_id]}.png')
    label = preproc_image(label_file)

    # Changing scale for the 'No label' encoded as 255
    label[label == 255] = 4

    # Scaler for raw images between 0 and 1
    image = image / 255

    if use_mask:
        mask = load_mask(im_id, df)
        image = image * mask
        label = label * mask

    return [image, label]



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
        [image, label] = load_preproc(i, df)

        X.append(image)
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    print('✅ loaded raw images and labels')
    y_decomp = decompose_label(y)
    print('✅ decomposed labels into binary masks')

    return X, y_decomp


if __name__ == '__main__':

    small_dataset = 100

    df_train = create_df(MASK_PATH_TRAIN)
    df_train = df_train.iloc[:small_dataset]

    X_train, y_train = load_images(df_train)
    print(X_train.shape, y_train.shape)
