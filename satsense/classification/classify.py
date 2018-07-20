"""Implementation of classification"""
import configparser
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, matthews_corrcoef

from satsense.image import SatelliteImage
from satsense.classification import Mask, Dataset
# Not occuring in the code, but can be specified in the .ini file
from satsense.bands import WORLDVIEW3

config = configparser.ConfigParser()
config.read("classify.ini")

# Image settings
IMAGE_FOLDER = config['Image']['folder']
BANDS = eval(config['Image']['bands'])
TRAIN_IMAGES = [n.strip() for n in config['Image']['train_images'].split(",")]
TEST_IMAGE = config['Image']['test_image'].strip()

# Feature settings
TILE_SIZE = eval(config['Features']['tile_size'])
THRESHOLD = eval(config['Features']['threshold'])
FEATURE_FOLDER = config['Features']['folder']
MASK_FOLDER = config['Masks']['folder']

LABELS = {
    'BUILDING': 1,
    'SLUM': 2,
    'VEGETATION': 3
}

def load_masks(image_name):
    """ Load the vegetation and slum mask from the mask folder specified in
        classify.ini. """
    path = os.path.join(MASK_FOLDER, "vegetation", image_name + ".npy")
    vegetation_mask = Mask.load_from_file(path)
    path = os.path.join(MASK_FOLDER, "slum", image_name + ".npy")
    slum_mask = Mask.load_from_file(path)
    path = os.path.join(MASK_FOLDER, "building", image_name + ".npy")
    building_mask = Mask.load_from_file(path)
    
    # We resample the masks from block of (1x1) pixels to (TILE_SIZExTILE_SIZE)
    # in order to get the same shape as the feature vector
    slum_mask = slum_mask.resample(TILE_SIZE, THRESHOLD)
    vegetation_mask = vegetation_mask.resample(TILE_SIZE, THRESHOLD)
    building_mask = building_mask.resample(TILE_SIZE, THRESHOLD)

    return slum_mask, vegetation_mask, building_mask

def load_feature_vector(image_name):
    return np.load(os.path.join(FEATURE_FOLDER, image_name + ".npy"))


def create_training_set():
    X_train = None
    y_train = None
    for imagefile in TRAIN_IMAGES:
        image_name = os.path.splitext(imagefile)[0]

        slum_mask, vegetation_mask, building_mask = load_masks(image_name)
        feature_vector = load_feature_vector(image_name)
          
    
        X_0, y_0 = Dataset(feature_vector).createXY(building_mask,
                                                    in_label=LABELS['BUILDING'])
        X_1, y_1 = Dataset(feature_vector).createXY(slum_mask,
                                                    in_label=LABELS['SLUM'])
        X_2, y_2 = Dataset(feature_vector).createXY(vegetation_mask,
                                                    in_label=LABELS['VEGETATION'])

        if X_train is None:
            X_train = np.concatenate((X_0, X_1, X_2), axis=0)
        else:
            X_train  = np.concatenate((X_train, X_0, X_1, X_2), axis=0)
        
        if y_train is None:
            y_train = np.concatenate((y_0, y_1, y_2), axis=0)
        else:
            y_train = np.concatenate((y_train, y_0, y_1, y_2), axis=0)
    
    return X_train, y_train

def create_test_set():
    image_name = os.path.splitext(TEST_IMAGE)[0]
    slum_mask, vegetation_mask, building_mask = load_masks(image_name)
    feature_vector = load_feature_vector(image_name)

    y_test = np.full(feature_vector.shape[:2], LABELS['BUILDING'])
    y_test[slum_mask == 1] =  LABELS['SLUM']                                                     
    y_test[vegetation_mask == 1] =  LABELS['VEGETATION']

    nrows = feature_vector.shape[0] * feature_vector.shape[1]
    nfeatures = feature_vector.shape[2]

    X_test = np.reshape(feature_vector, (nrows, nfeatures))
    y_test = np.reshape(y_test, (nrows, ))

    return X_test, y_test, feature_vector.shape[:2]

if __name__ == "__main__":
    print("Creating training set...")
    X_train, y_train = create_training_set()
    # print("Oversampling")
    # print(np.unique(y_train, return_counts=True))

    # ratio = {LABELS["BUILDING"]: np.count_nonzero(y_train == LABELS["BUILDING"]),
    #          LABELS["VEGETATION"]:  np.count_nonzero(y_train == LABELS["VEGETATION"]),
    #          LABELS["SLUM"]: np.count_nonzero(y_train == LABELS["BUILDING"]) + np.count_nonzero(y_train == LABELS["VEGETATION"])}
             
    # print(ratio)
    # X_train, y_train = RandomOverSampler(ratio=ratio).fit_sample(X_train, y_train)
    print("Creating test set...")
    X_test, y_test, original_shape = create_test_set()

    classifier = GradientBoostingClassifier(verbose=True)
    print("fitting...")

    # with open('gb_s2_b10.pickle', 'rb') as f:
    #     classifier = pickle.load(f)

    classifier.fit(X_train, y_train)

    with open('gb_s3_b10_smt.pickle', 'wb') as f:
        pickle.dump(classifier, f)

    y_pred = classifier.predict(X_test)
    # Label the vegetation as buildings to create more accurate representation of the performance
    y_pred[y_pred == LABELS['VEGETATION']] = LABELS['BUILDING']
    y_test[y_test == LABELS['VEGETATION']] = LABELS['BUILDING']
    
    print(np.unique(y_pred, return_counts=True))
    print(np.unique(y_test, return_counts=True))

    print(matthews_corrcoef(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    result_mask = Mask(np.reshape(y_pred, original_shape))
    path = os.path.join(IMAGE_FOLDER, TEST_IMAGE)
    test_image = SatelliteImage.load_from_file(path, BANDS)
    result_mask.overlay(test_image.rgb)
    plt.savefig('result')
    plt.show()
