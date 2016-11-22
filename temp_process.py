#!/usr/bin/env python
from numpy import arange, genfromtxt, zeros, mean, float32, sum, row_stack, array, exp
from scipy.signal import chirp, convolve
from scipy import fft
from peakutils import indexes, baseline
import math


class TempertureFeature:
    def __init__(self, temp=0):
        self.temperture = []
        self.temperture.append(temp)
        pass

    def concatenate_features(self, temp_2_concatenate):
        if temp_2_concatenate is None:
            return
        self.temperture.append(temp_2_concatenate)
        pass

    def calc_feature_distance(self, feature_to_compare, matches, descriptor_match_normalize_factor=1):
        # Initializing vecs
        all_distances = []
        all_scale_ratio = []
        self_matched_features_idx = []
        # Todo: calc with lambda
        for m in matches:
            #add Distance to all_distance vec
            all_distances.append(m.distance)
            #Normalize by 1
            if m.distance > descriptor_match_normalize_factor:
                descriptor_match_normalize_factor = m.distance
            # scale ratio - give distance his chance :)
            s1 = self.temperture[m.queryIdx]
            s2 = feature_to_compare.temperture[m.trainIdx]
            # ratio = exp(-0.5*((s1-s2)/0.5)**2)
            # all_scale_ratio.append(ratio)
            if s1 > s2:
                all_scale_ratio.append(s2 / s1)
            else:
                all_scale_ratio.append(s1 / s2)
            self_matched_features_idx.append(m.queryIdx)
        # descriptor distances
        norm_mean_descriptor_dist = sum(array(all_distances)) / (len(all_distances) * descriptor_match_normalize_factor)
        # scale ratio distance
        norm_scale_ratio_distance = 1 - sum(array(all_scale_ratio)) / (len(all_scale_ratio))
        # num of matches distance
        norm_num_matches_dist = 1 - float(len(matches)) / float(len(self.temperture))
        distance = self.get_weighted_distance(norm_mean_descriptor_dist, norm_num_matches_dist,
                                              norm_scale_ratio_distance)
        # distance = (norm_mean_descriptor_dist + norm_num_matches_dist + norm_scale_ratio_distance) / 3
        if math.isnan(distance):
            pass
        return distance

    def get_weighted_distance(self, norm_mean_descriptor_dist, norm_num_matches_dist, norm_scale_ratio_distance):
        descriptor_dist_w = 0.2
        matches_dist_w = 0.5
        ratio_distance = 0.3
        return descriptor_dist_w * norm_mean_descriptor_dist + \
               matches_dist_w * norm_num_matches_dist + ratio_distance * norm_scale_ratio_distance