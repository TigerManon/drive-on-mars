import os

# Local setup

MODEL_TARGET = os.environ.get("MODEL_TARGET")

GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")


# Customize your path here
PACKAGE_PATH = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(PACKAGE_PATH, 'raw_data', 'ai4mars-dataset-merged-0.1', 'msl')

IMAGE_PATH = os.path.join(DATA_PATH, 'images', 'edr')
MASK_PATH_TRAIN = os.path.join(DATA_PATH, 'labels', 'train')
MASK_PATH_TESTS = os.path.join(DATA_PATH, 'labels', 'test')
TESTS_DIR = ['masked-gold-min1-100agree', 'masked-gold-min2-100agree', 'masked-gold-min3-100agree']

MASK_ROVER = os.path.join(DATA_PATH, 'images', 'mxy')
RANGE_30M = os.path.join(DATA_PATH, 'images', 'rng-30m')

LOCAL_REGISTRY_PATH = os.path.join(PACKAGE_PATH, 'raw_data')


labels_key = {
    0: 'soil',
    1: 'bedrock',
    2: 'sand',
    3: 'big rock',
    4: '(no label)',
}

RESIZE_SHAPE = 256
