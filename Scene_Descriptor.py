import cv2
import Image_Features
import math
import numpy as np

__author__ = 'rons'


class SceneDescriptor:
    def __init__(self, image_path, capture_data1, capture_data2=None, features_list=["SIFT"]):

        # class structure:
        # capture_data1 : CaptureData class, that contain the rgb image sonar image and theta angle
        # capture_data2 : [optional]CaptureData class, that contain the second rgb image sonar image and theta angle
        # features_list : List of the features in self.features
        # features      : Dict with keys from features_list and ImageFeature class values,
        # that contain features from an rgb image.

        image1 = cv2.imread(image_path + capture_data1.rgb_image_path)
        sonar1 = cv2.imread(image_path + capture_data1.sonar_image_path)
        if capture_data2 is not None:
            image2 = cv2.imread(image_path + capture_data2.rgb_image_path)
            sonar2 = cv2.imread(image_path + capture_data1.sonar_image_path)
        else:
            image2 = None
            sonar2 = None
        self.capture_data1 = capture_data1
        self.capture_data2 = capture_data2
        image_features = Image_Features.ImageFeature
        feature_dict = {}
        self.features = {}
        self.features_list = features_list
        for feature_type in self.features_list:
            if "Sonar".upper() not in feature_type.upper():
                feature_dict[feature_type] = image_features(feature_type, image1, image2)
            else:
                sonar_ind = feature_type.find("_")
                #feature_type = feature_type[:sonar_ind]
                feature_dict[feature_type] = image_features(feature_type[:sonar_ind], sonar1, sonar2)
        self.features = feature_dict

    def get_scenes_distance(self, scene2compare):

        feature_counter = 0
        dist_by_feature_type = {}
        total_distance = 0
        for feature_type in self.features_list:
            if feature_type in scene2compare.features_list:
                feature_counter += 1
                curr_feature = self.features[feature_type]
                dist, dist_by_type = curr_feature.compare_with_feature(scene2compare.features[feature_type])
                dist_by_feature_type[feature_type] = dist_by_type
                total_distance += dist

        if feature_counter == 0:
            total_distance = np.inf
        else:
            total_distance = total_distance / feature_counter
        if math.isnan(total_distance):
            pass
        return total_distance, dist_by_feature_type

    def get_sonar_scenes_distance(self, scene2compare):

        feature_counter = 0
        dist_by_feature_type = {}
        total_distance = 0
        for feature_type in self.features_list:
            if "SONAR".upper() in feature_type.upper():
                if feature_type in scene2compare.features_list:
                    feature_counter += 1
                    curr_feature = self.features[feature_type]
                    dist, dist_by_type = curr_feature.compare_with_feature(scene2compare.features[feature_type])
                    dist_by_feature_type[feature_type] = dist_by_type
                    total_distance += dist

        if feature_counter == 0:
            total_distance = np.inf
        else:
            total_distance = total_distance / feature_counter
        if math.isnan(total_distance):
            pass
        return total_distance, dist_by_feature_type








