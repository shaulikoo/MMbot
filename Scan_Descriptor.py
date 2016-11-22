import numpy as np
import Scene_Descriptor
import math
import Image_Features
from Run_On_All_Images import ScanData

__author__ = 'rons'


class ScanDescriptor:
    SCENE_SINGLE_IMAGE = 0
    SCENE_IMAGE_PAIR = 1

    def __init__(self, image_folder_path, scan_data, scene_feature_type=SCENE_SINGLE_IMAGE, features_list=["SIFT"]):
        self.scene_feature_type = scene_feature_type
        self.scan_data = []
        self.update_scan_data(scan_data)
        self.scene_descriptors_dict = {}
        self.thetas = []
        scene_descriptors_dict = {}
        for c_data in self.scan_data.capture_data:
            try:
                theta = float(c_data[0].theta_in_deg + c_data[1].theta_in_deg) / 2
                self.thetas.append(theta)
                scene_descriptors_dict[theta] = Scene_Descriptor.SceneDescriptor(image_folder_path, c_data[0],
                                                                                 c_data[1], features_list=features_list)
            except:
                self.thetas.append(c_data.theta_in_deg)
                scene_descriptors_dict[c_data.theta_in_deg] = Scene_Descriptor.SceneDescriptor(image_folder_path,
                                                                                               c_data,
                                                                                               features_list=features_list)
        self.scene_descriptors_dict = scene_descriptors_dict

    def update_scan_data(self, scan_data):
        tmp_scan_data = ScanData(scan_data)
        if self.scene_feature_type == ScanDescriptor.SCENE_SINGLE_IMAGE:
            self.scan_data = tmp_scan_data
        else:
            captures = tmp_scan_data.capture_data
            capture_pairs = [captures[i:i + 2] for i in xrange(0, len(captures), 2)]
            tmp_scan_data.capture_data = capture_pairs
            self.scan_data = tmp_scan_data

    def get_scan_distance_theta_vs_theta(self, scan2compare):

        scenes_counter = 0
        dist_by_theta = {}
        total_distance = 0

        for curr_theta in self.thetas:
            if curr_theta in scan2compare.thetas:
                scenes_counter += 1
                curr_scene_descriptor = self.scene_descriptors_dict[curr_theta]
                total_scene_distance, scene_dist_by_feature_type = curr_scene_descriptor.get_scenes_distance \
                    (scan2compare.scene_descriptors_dict[curr_theta])
                dist_by_theta[curr_theta] = scene_dist_by_feature_type
                total_distance += total_scene_distance
        if scenes_counter == 0:
            total_distance = np.inf
        else:
            total_distance = total_distance / scenes_counter
        if math.isnan(total_distance):
            pass
        return total_distance, dist_by_theta

    def merge_all_scene_features(self, is_only_sonar):
        all_features = {}
        for theta in self.thetas:
            curr_scene_descriptor = self.scene_descriptors_dict[theta]
            for feature_type in curr_scene_descriptor.features_list:
                if is_only_sonar and "SONAR".upper() not in feature_type.upper():
                    continue
                try:
                    all_features[feature_type].concatenate_features(curr_scene_descriptor.features[feature_type])
                except:  # if the first feature type found
                    try:
                        im_feature = Image_Features.ImageFeature(curr_scene_descriptor.features[feature_type])
                        all_features[feature_type] = im_feature
                    except:
                        pass
        return all_features

    def get_scan_distance_all_thetas(self, scan2compare, is_only_sonar=False):

        all_images_feature = self.merge_all_scene_features(is_only_sonar)
        all_images_feature_2_compare = scan2compare.merge_all_scene_features(is_only_sonar)
        f_counter = 0
        total_dist = 0
        dist_by_type = {}
        for key in all_images_feature.keys():
            distance, dist_by_type[key] = all_images_feature[key].compare_with_feature(
                all_images_feature_2_compare[key])
            total_dist += distance
            f_counter += 1
        if f_counter == 0:
            total_dist = np.inf
        else:
            total_dist /= f_counter
        return total_dist, dist_by_type


