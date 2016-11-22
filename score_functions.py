__author__ = 'Shaul'
from Image_Visual_Features import ImageVisualFeatures
import numpy as np
import time
import Features_Extractor
from sonar_process import SonarFeature
from cv2 import BFMatcher
from temp_process import TempertureFeature

twister = [180, 162, 144, 126, 108, 90, 72, 54, 36, 18, 0, -18, -36, -54, -72, -90, -108, -126, -144, -162, -180]
# twister = [180, 90, 0, -90]


def score_sonar_by_fetures(DM_ref, DM_check, min_matches_needed=0):
    scan_descriptors_1 = DM_ref.scan_descriptors
    scan_descriptors_2 = DM_check.scan_descriptors
    first_ref = True
    first_check = True
    for angle in twister:
        for fet in scan_descriptors_1[angle].sonar.features:
            if first_ref:
                concatenate_feature_1 = SonarFeature(fet.distance[0], fet.fft)
                first_ref = False
            else:
                concatenate_feature_1.concatenate_features(fet)
        for fet in scan_descriptors_2[angle].sonar.features:
            if first_check:
                concatenate_feature_2 = SonarFeature(fet.distance[0], fet.fft)
                first_check = False
            else:
                concatenate_feature_2.concatenate_features(fet)
    matches = knn_matcher(concatenate_feature_1, concatenate_feature_2, nndist=0.85)
    if len(matches) < min_matches_needed * len(concatenate_feature_1.fft) or not matches:
        # if not enough matches return inf
        # score = np.inf
        return 0
    distance = concatenate_feature_1.calc_feature_distance(concatenate_feature_2, matches)
    score = 1 - distance
    return score


def score_vision(DM_ref, DM_check, min_matches_needed=0):
    # distance of DM2 from DM1
    scan_descriptors_1 = DM_ref.scan_descriptors
    scan_descriptors_2 = DM_check.scan_descriptors
    first = True
    for angle in twister:
        if first:
            concatenate_feature_1 = ImageVisualFeatures(scan_descriptors_1[angle].vision)
            concatenate_feature_2 = ImageVisualFeatures(scan_descriptors_2[angle].vision)
            first = False
        else:
            concatenate_feature_1.concatenate_features(scan_descriptors_1[angle].vision)
            concatenate_feature_2.concatenate_features(scan_descriptors_2[angle].vision)
            if len(concatenate_feature_1.feature) != concatenate_feature_1.descriptor.shape[0]:
                print "error"
    matches = []
    dist_by_type = {}
    if concatenate_feature_1.feature_type == "SIFT":
        # Match descriptors with 2 nearest neighbours
        fe = Features_Extractor.FeaturesExtractor
        matches = fe.sift_matcher(concatenate_feature_1.descriptor, concatenate_feature_2.descriptor)

    if len(matches) < min_matches_needed * len(concatenate_feature_1.feature) or not matches:
        # if not enough matches return inf
        # score = np.inf
        score = 0
        dist_by_type["matches"] = np.inf
    else:
        pass
        distance, dist_by_type = concatenate_feature_1.calc_feature_distance(concatenate_feature_2, matches)
    # score = concatenate_feature_1.calc_feature_distance(concatenate_feature_2)
    score = 1 - distance
    return score


def score_Temperture(DM1, DM2, min_matches_needed=0):
    # distance of DM2 from DM1
    scan_descriptors_1 = DM1.scan_descriptors
    scan_descriptors_2 = DM2.scan_descriptors
    first = True
    for angle in twister:
        if first:
            concatenate_feature_1 = TempertureFeature(scan_descriptors_1[angle].temp)
            concatenate_feature_2 = TempertureFeature(scan_descriptors_2[angle].temp)
            first = False
        else:
            concatenate_feature_1.concatenate_features(scan_descriptors_1[angle].temp)
            concatenate_feature_2.concatenate_features(scan_descriptors_2[angle].temp)
    score_rat = (np.sqrt((np.asarray(concatenate_feature_1.temperture)-np.asarray(concatenate_feature_2.temperture))**2))
    score_rat /= max(score_rat)
    return 1 - score_rat.mean()


def calc_1_vs_all_scores_by_feature(map, sample, e='all'):
    if (e=='vision'):
        curr_dm_scores_vision = np.zeros(len(map))
        for dm_2_comp in map:
            curr_dm_scores_vision[dm_2_comp.scan_number-1] = score_vision(sample, dm_2_comp)
        return curr_dm_scores_vision
    if (e=='sonar'):
        curr_dm_scores_sonar = np.zeros(len(map))
        for dm_2_comp in map:
            curr_dm_scores_sonar[dm_2_comp.scan_number-1] = score_sonar_by_fetures(sample, dm_2_comp)
        return curr_dm_scores_sonar
    if (e=='temp'):
        curr_dm_scores_temp = np.zeros(len(map))
        for dm_2_comp in map:
            curr_dm_scores_temp[dm_2_comp.scan_number-1] = score_Temperture(sample, dm_2_comp)
        return curr_dm_scores_temp
    if (e=='all'):
        # curr_dm_scores_temp = np.zeros(len(map))
        curr_dm_scores_vision = np.zeros(len(map))
        curr_dm_scores_sonar = np.zeros(len(map))
        for dm_2_comp in map:
            curr_dm_scores_vision[dm_2_comp.scan_number-1] = score_vision(sample, dm_2_comp)
            curr_dm_scores_sonar[dm_2_comp.scan_number-1] = score_sonar_by_fetures(sample, dm_2_comp)
            # curr_dm_scores_temp[dm_2_comp.scan_number-1] = score_Temperture(sample, dm_2_comp)
        return curr_dm_scores_vision, curr_dm_scores_sonar #,curr_dm_scores_temp


def fusion(scores_vision, scores_sonar, bayes):
    if len(scores_vision) != len(scores_sonar):
        raise NameError('Scores Not the same length')
    bayes.bayesian_fusion(scores_sonar, scores_vision)
    fusion_score = bayes.predict
    # for pos in range(0, len(scores_vision)):
    #     fusion_score.append(0.5*scores_vision[pos] + 0.5*scores_sonar[pos])
    return fusion_score


def knn_matcher(descriptors1, descriptors2, nndist=0.75):
    bf = BFMatcher()
    # Match descriptors with 2 nearest neighbours
    matches = bf.knnMatch(descriptors1.fft, descriptors2.fft, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < nndist * n.distance:
            good_matches.append(m)
    return good_matches


class bayes_fusion:
    def __init__(self, length):
        self.predict = None
        self.perception = None
        self.z = None
        self.eye = np.eye(length)
        self.noise_vision_mag = 0.02
        self.noise_sonar_mag = 0.05
        self.R = self.noise_vision_mag ** 2 * self.eye
        self.Q = self.noise_sonar_mag ** 2 * self.eye
        self.P = self.R
        self.K = None

    # def bayesian_fusion(self, sonar_score, vision_score):
    #     self.predict = vision_score
    #     self.P += self.R
    #     self.z = sonar_score
    #     self.K = self.P.dot(self.eye).dot(np.linalg.inv(self.eye.dot(self.P).dot(self.eye)+self.Q))
    #     self.predict = self.predict + self.K.dot(self.z-self.eye.dot(self.predict))
    #     self.P = (self.eye-self.K.dot(self.eye)).dot(self.P)

    def bayesian_fusion(self, sonar_score, vision_score):
        t0 = 0.8
        predict = (sonar_score*vision_score)/(sonar_score*vision_score + (1-sonar_score)*(1-vision_score))
        for i in range(0,len(predict)):
            if predict[i] > t0:
                predict[i] = 1
            elif (predict[i] >= 0.5) and (predict[i] <= t0):
                print predict[i]
                predict[i] = (predict[i] + t0 - 1)/(2*t0 -1)
                print predict[i]
            else:
                predict[i] = predict[i]
        self.predict = predict
        # self.predict = vision_score
        # self.P += self.R
        # self.z = sonar_score
        # self.K = self.P.dot(self.eye).dot(np.linalg.inv(self.eye.dot(self.P).dot(self.eye)+self.Q))
        # self.predict = self.predict + self.K.dot(self.z-self.eye.dot(self.predict))
        # self.P = (self.eye-self.K.dot(self.eye)).dot(self.P)


class Kalman_filter:
    def __init__(self):
        self.predict = None
        self.perception = None
        self.z = None
        self.map_odom_noise = 0.05
        self.dm_odom_noise = 0.05
        self.R = self.map_odom_noise ** 2 * np.eye(2)
        self.Q = self.dm_odom_noise ** 2 * np.eye(2)
        self.P = self.R
        self.K = None

    def Kalman_filter(self, map_odom, dm_odom):
        self.predict = map_odom[:2]
        self.P += self.R
        self.z = dm_odom[:2]
        self.K = self.P.dot(np.eye(2)).dot(np.linalg.inv(np.eye(2).dot(self.P).dot(np.eye(2))+self.Q))
        self.predict = self.predict + self.K.dot(self.z-np.eye(2).dot(self.predict))
        self.P = (np.eye(2)-self.K.dot(np.eye(2))).dot(self.P)