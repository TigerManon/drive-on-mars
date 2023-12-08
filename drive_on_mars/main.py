###########
# Imports #
###########

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import cv2
from PIL import Image
from typing import Tuple
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models as sm
from segmentation_models import get_preprocessing
from segmentation_models.losses import bce_jaccard_loss
from segmentation_models.metrics import iou_score
from tensorflow.keras import callbacks
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda

from drive_on_mars.model.model import initialize_model, compile_model, train_model
from drive_on_mars.model.registry import save_model, save_results, load_model


#####################
# Data & Preprocess #
#####################

def preprocess():
    pass









    print("✅ preprocess done")


##################
# Model Training #
##################

def model_training():
    """
    - Create Train Test Split
    - Train the model
    - Save the model
    - Compute & save a validation performance metric
    """
    # Train Test Val Split - Create X_train, X_test, y_train, y_test, X_val, y_val #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = 0.3, random_state = 42)

    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, test_size = 0.5, random_state = 42)  # test=15%

    '''
    ?????????????????????
    # Create X_train_processed using '??DATA??.py'
    X_train_processed = preprocess(X_train)
    ?????????????????????
    '''

    # Train the model on the training set, using 'model.py' #
    model = None
    learning_rate = 0.0005
    batch_size = 256
    patience = 2

    model = initialize_model(input_shape=X_train.shape[1:])
    model = compile_model(model, learning_rate=learning_rate)
    model, history = train_model(model,
                                X_train,
                                y_train,
                                batch_size=batch_size,
                                patience=patience,
                                validation_data=None, # overrides validation_split
                                validation_split=0.3)

###########
# Scoring #
###########

    # Compute the validation metric (min iou_score)
    val_iou_score = np.min(history.history['iou_score'])

##########################
# Saving model & results #
##########################

    params = dict(
        learning_rate=learning_rate,
        batch_size=batch_size,
        patience=patience
    )

    save_model(model=model)
    save_results(params=params, metrics=dict(iou_score=val_iou_score))

    print("✅ model_training() done")


def pred(X_test):
    """
    Load the model & make a prediction
    """
    model = load_model()
    X_processed = preprocess_features(X_pred)
    y_pred = model.predict(X_processed)

    print(f"✅ pred() done")




if __name__ == '__main__':
    preprocess()
    model_training()
    pred()
    save_model()
