import Features_Extractor
import numpy as np
import math

__author__ = 'rons'


class ImageFeature:
    def __init__(self, feature_type, image1=None, image2=None):
        try:
            if isinstance(feature_type, ImageFeature):
                self.dup_image_feature(feature_type)
            else:
                self.feature_type = feature_type
                self.descriptor = []
                self.weights = []
                self.feature = []
                self.update_obj_params_from_images(image1, image2)

        except Exception as e:
            print ("Failed init ImageFeature for " + feature_type + " features")
            print (e.message)
            pass

    def dup_image_feature(self, image_feature2dup):
        self.feature_type = image_feature2dup.feature_type
        self.descriptor = image_feature2dup.descriptor
        self.weights = image_feature2dup.weights
        self.feature = image_feature2dup.feature

    def update_obj_params_from_images(self, image1, image2):
        if image1 is not None:
            features_extractor = Features_Extractor.FeaturesExtractor
            kp = []
            descriptors = []
            if self.feature_type == "SIFT":
                if image2 is not None:  # we have image pair
                    kp_pairs, descriptors_pairs, matches = \
                        features_extractor.get_matched_sift_points_and_features_from_image_pair(image1, image2)
                    kp, descriptors = \
                        ImageFeature.keep_points_features_from_image_pair(self, kp_pairs, descriptors_pairs, matches)
                else:
                    kp, descriptors = \
                        features_extractor.get_sift_points_and_features(image1)
            self.feature = kp
            self.descriptor = descriptors
            self.weights = [float(1) for i in range(len(kp))]

    def concatenate_features(self, image_feature_2_concatenate):
        self.feature += image_feature_2_concatenate.feature
        self.weights += image_feature_2_concatenate.weights
        self.descriptor = np.concatenate([self.descriptor, image_feature_2_concatenate.descriptor])
        pass

    def keep_points_features_from_image_pair(self, kp_pairs, descriptors_pairs, matches, keep_image_points=2,
                                             min_dist=0, max_aloud_scale_diff_precent=0.1):

        kp = []
        descriptors = []
        if self.feature_type == "SIFT":
            for kp_ind in range(len(kp_pairs[0])):
                kp1 = kp_pairs[0][kp_ind]
                kp2 = kp_pairs[1][kp_ind]
                scale1 = kp1.size
                scale2 = kp2.size
                if scale1 > scale2:
                    scale_ratio = scale2 / scale1
                else:
                    scale_ratio = scale1 / scale2
                distance = matches[kp_ind].distance
                if scale_ratio > 1 - max_aloud_scale_diff_precent and distance > min_dist:
                    # keep one point from pair
                    kp_pairs[keep_image_points - 1][kp_ind].size = (scale1 + scale2) / 2
                    kp.append(kp_pairs[keep_image_points - 1][kp_ind])
                    # average the descriptor
                    descriptors.append((descriptors_pairs[0][kp_ind] + descriptors_pairs[1][kp_ind]) / 2)
                kp_ind += 1
            descriptors = np.array(descriptors)
            pass
        return kp, descriptors

    def compare_with_feature(self, feature_to_compare, min_matches_needed=0):
        dist_by_type = {}
        if feature_to_compare.feature_type == self.feature_type:
            my_descriptors = self.descriptor
            descriptor2 = feature_to_compare.descriptor
            pass
        else:
            print "not same feature, need to be same feature to be able to compare"
            return np.inf
        matches = []
        if self.feature_type == "SIFT":
            # Match descriptors with 2 nearest neighbours
            fe = Features_Extractor.FeaturesExtractor
            matches = fe.sift_matcher(my_descriptors, descriptor2)

        if len(matches) < min_matches_needed * len(self.feature) or not matches:
            # if not enough matches return inf
            distance = np.inf
            dist_by_type["matches"] = np.inf
        else:
            distance, dist_by_type = ImageFeature.calc_feature_distance(self, feature_to_compare, matches)
        return distance, dist_by_type

    def calc_feature_distance(self, feature_to_compare, matches, descriptor_match_normalize_factor=1):
        distance = np.inf
        dist_by_type = {}
        if self.feature_type == "SIFT":
            all_distances = []
            all_scale_ratio = []
            self_matched_features_idx = []
            weights = []
            # Todo: calc with lambda
            for m in matches:
                all_distances.append(m.distance)
                if m.distance > descriptor_match_normalize_factor:
                    descriptor_match_normalize_factor = m.distance
                # scale ratio
                s1 = self.feature[m.queryIdx].size
                s2 = feature_to_compare.feature[m.trainIdx].size
                if s1 > s2:
                    all_scale_ratio.append(s2 / s1)
                else:
                    all_scale_ratio.append(s1 / s2)
                self_matched_features_idx.append(m.queryIdx)
                weights.append(self.weights[m.queryIdx])
            # descriptor_match_normalize_factor = max(matches, key=lambda x:x.distance)

            # descriptor distances
            norm_mean_descriptor_dist = np.sum(np.array(all_distances) * np.array(weights)) / (
            len(all_distances) * descriptor_match_normalize_factor)
            # scale ratio distance
            norm_scale_ratio_distance = 1 - np.sum(np.array(all_scale_ratio) * np.array(weights)) / (
            len(all_scale_ratio))
            # num of matches distance
            norm_num_matches_dist = 1 - float(len(matches)) / float(len(self.feature))
            dist_by_type["descriptors"] = norm_mean_descriptor_dist
            dist_by_type["scale"] = norm_scale_ratio_distance
            dist_by_type["matches"] = norm_num_matches_dist
            distance = (norm_mean_descriptor_dist + norm_num_matches_dist + norm_scale_ratio_distance) / 3
            if math.isnan(distance):
                pass
            self.update_weights(distance, self_matched_features_idx)
        return distance, dist_by_type

    def update_weights(self, distance, self_matched_features_idx, distance_threshold=0.15):
        if distance > distance_threshold:
            return
        else:
            for f_idx in range(len(self.feature)):
                if f_idx in self_matched_features_idx:
                    self.weights[f_idx] *= 2
                else:
                    self.weights[f_idx] /= 2
