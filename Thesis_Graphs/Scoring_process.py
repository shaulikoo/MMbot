__author__ = 'Shaul'
import sonar_process as sp
import simplejson as sj
import numpy as np
import cv2
import math
from Image_Visual_Features import ImageVisualFeatures
from heatMap_class import heatMap
import pickle_functions
import time
import re
import matplotlib.pyplot as plt
from extract_tracker import tracker


class Tools:
    def __init__(self):
        pass

    def genrate_huge_map(self, pickle_name, map_size=20):
        map_array = []
        for i in range(1,map_size+1):
            print 'Start Gen for JSON%s'%i
            timer = time.time()
            map_array.append(Node(dataLoc + '/scan%s.json'%i))
            print "Finished json%s in %s seconds"%(i, time.time()-timer)
        pickle_functions.save_var(map_array, pickle_name)
        return 0


class PFNInstance:
    def scoring_process(self, dm1, dm2, sonar_instance_bool, min_matches_needed=0):
        if sonar_instance_bool:
            matches = self.knn_matcher(dm1.fftPFN, dm2.fftPFN, nndist=0.85)
            if len(matches) < min_matches_needed * len(dm1.fftPFN) or not matches:
                return 0
            distance = dm1.calc_feature_distance(dm2, matches)

        else:
            matches = self.knn_matcher(dm1.descriptor, dm2.descriptor, nndist=0.85)
            if len(matches) < min_matches_needed * len(dm1.feature) or not matches:
                return 0
            distance, dist_by_type = dm1.calc_feature_distance(dm2, matches)
        score = 1 - distance
        return score

    @staticmethod
    def get_weighted_distance(norm_mean_descriptor_dist, norm_num_matches_dist, norm_scale_ratio_distance):
        descriptor_dist_w = 0.2
        matches_dist_w = 0.5
        ratio_distance = 0.3
        return descriptor_dist_w * norm_mean_descriptor_dist + \
               matches_dist_w * norm_num_matches_dist + ratio_distance * norm_scale_ratio_distance

    @staticmethod
    def knn_matcher(descriptors1, descriptors2, nndist=0.75, k_val = 2):
        bf = cv2.BFMatcher()
        # Match descriptors with 2 nearest neighbours
        matches = bf.knnMatch(descriptors1, descriptors2, k = k_val)
        if k_val > 1:
            good_matches = []
            for m, n in matches:
                if m.distance < nndist * n.distance:
                    good_matches.append(m)
        else:
            good_matches = matches
        return good_matches


class Sonar(PFNInstance):
    def __init__(self, jsonfile):
        first_all = True
        first_forward = True
        first_right = True
        self.all_features_angle = {}
        for data in jsonfile:
            feature_extraction = sp.SonarSignal(dataLoc + data['capture']['Sonar'])
            angle = data['angle']
            self.all_features_angle[str(angle)] = feature_extraction
            for feature in feature_extraction.features:
                if first_all:
                    self.distancePFN = []
                    self.fftPFN = feature.fft.reshape(1, 60)
                    self.distancePFN.append(feature.distance)
                    first_all = False
                else:
                    self.concatenate_features(feature)
                if first_forward and angle == 0:
                    self.FdistancePFN = []
                    self.FfftPFN = feature.fft.reshape(1, 60)
                    self.FdistancePFN.append(feature.distance)
                    first_forward = False
                elif angle == 0:
                    self.Fconcatenate_features(feature)
                if first_right and angle == 18:
                    self.RdistancePFN = []
                    self.RfftPFN = feature.fft.reshape(1, 60)
                    self.RdistancePFN.append(feature.distance)
                    first_right = False
                elif angle == 18:
                    self.Rconcatenate_features(feature)

    def concatenate_features(self, feature):
            self.fftPFN = np.row_stack((self.fftPFN, feature.fft.reshape(1, 60)))
            self.distancePFN.append(feature.distance)

    def Fconcatenate_features(self, feature):
            self.FfftPFN = np.row_stack((self.FfftPFN, feature.fft.reshape(1, 60)))
            self.FdistancePFN.append(feature.distance)

    def Rconcatenate_features(self, feature):
            self.RfftPFN = np.row_stack((self.RfftPFN, feature.fft.reshape(1, 60)))
            self.RdistancePFN.append(feature.distance)

    def calc_feature_distance(self, feature_to_compare, matches, descriptor_match_normalize_factor=1):
        # Initializing vecs
        knn_distances = []
        distnace_change_value = []
        for m in matches:
            #add Distance to all_distance vec
            knn_distances.append(m.distance)
            #Normalize by 1
            if m.distance > descriptor_match_normalize_factor:
                descriptor_match_normalize_factor = m.distance
            # scale ratio - give distance his chance :)
            s1 = self.distancePFN[m.queryIdx][0]
            s2 = feature_to_compare.distancePFN[m.trainIdx][0]
            distnace_change = np.exp(-0.5*((s1-s2)/0.3)**2)
            # ratio = s2 / s1
            distnace_change_value.append(distnace_change)
        # descriptor distances
        norm_mean_descriptor_dist = sum(np.array(knn_distances)) / (len(knn_distances) * descriptor_match_normalize_factor)
        # scale ratio distance
        norm_distnace_change_value = 1 - sum(np.array(distnace_change_value)) / (len(distnace_change_value))
        # num of matches distance
        norm_num_matches_dist = 1 - \
                                ((float(len(matches)) / float(len(self.fftPFN)) + float(len(matches)) / float(len(feature_to_compare.distancePFN)))/2)
        distance = self.get_weighted_distance(norm_mean_descriptor_dist, norm_num_matches_dist,
                                              norm_distnace_change_value)
        # distance = (norm_mean_descriptor_dist + norm_num_matches_dist + norm_scale_ratio_distance) / 3
        if math.isnan(distance):
            pass
        return distance


class Vision(PFNInstance):
    def __init__(self, jsonfile):
        first = True
        for data in jsonfile:
            bgr_image = cv2.imread(dataLoc + data['capture']['Image'])
            feature_extraction_var = ImageVisualFeatures("SIFT", bgr_image)
            if first:
                self.visionPFN = ImageVisualFeatures(feature_extraction_var)
                first = False
            else:
                self.visionPFN.concatenate_features(feature_extraction_var)


class Odom:
    def __init__(self, jsonfile):
        self.x = None
        self.y = None
        self.phi = None
        self.extract_odom(jsonfile)

    def extract_odom(self, jsonfile):
        self.x = jsonfile[1]['capture']['odom_x']
        self.y = jsonfile[1]['capture']['odom_y']
        self.phi = jsonfile[1]['capture']['phi']


class Node(PFNInstance):
    def __init__(self, scanJSONpath):
        jsonfile = sj.loads(open(scanJSONpath).read())
        self.vision = Vision(jsonfile)
        self.sonar = Sonar(jsonfile)
        self.odom = Odom(jsonfile)
        self.node_number = int(re.findall('scan(.*?).json', scanJSONpath)[0])

    def __lt__(self, other):
        return self.node_number < other.node_number

    def bayesian_fusion(self, sonar_score, vision_score):
        t0 = 0.8
        bayesian = (sonar_score*vision_score)/(sonar_score*vision_score + (1-sonar_score)*(1-vision_score))
        if bayesian > t0:
            bayesian = 1
        elif (bayesian >= 0.5) and (bayesian <= t0):
            bayesian = (bayesian + t0 - 1)/(2*t0 -1)
        else:
            bayesian = bayesian
        return bayesian


class Map:
    def __init__(self):
        self.map_array = []

    def calculate_distance_by_odom(self, node, bias):
        odom = np.array((node.odom.x-bias[0], node.odom.y-bias[1]))
        odom = np.array([np.linalg.norm(odom),np.arctan2(odom[1],odom[0])])
        return odom

    def calculate_distance_by_sonar(self, node1, node2):
        matchesf = node1.sonar.knn_matcher(node1.sonar.FfftPFN, node2.sonar.FfftPFN, nndist=0.85, k_val=1)
        matchesr = node1.sonar.knn_matcher(node1.sonar.RfftPFN, node2.sonar.RfftPFN, nndist=0.85, k_val=1)
        a = node1.sonar.FdistancePFN[0][0]
        b = node1.sonar.RdistancePFN[0][0]
        c = np.sqrt(a**2.0 + b**2.0 -2.0*a*b*np.cos(np.deg2rad(18)))
        s = (a*b*np.sin(np.deg2rad(18)))/2.0
        h1 = (2.0*s)/c
        phi1 = np.rad2deg(np.arccos(h1/a))
        a1 = node2.sonar.FdistancePFN[0][0]
        b = node2.sonar.RdistancePFN[0][0]
        c = np.sqrt(a1**2.0 + b**2.0 -2.0*a1*b*np.cos(np.deg2rad(18)))
        s = (a1*b*np.sin(np.deg2rad(18)))/2.0
        h2 = (2.0*s)/c
        phi2 = np.rad2deg(np.arccos(h2/a1))
        return abs(a-a1)


        # # delta_y = node2.sonar.RdistancePFN[matchesr[0].trainIdx][0] - node1.sonar.RdistancePFN[matchesr[0].queryIdx][0]
        # distance = np.array((delta_x))
        # # distance = np.linalg.norm(delta)
        # return distance

    def add_to_map(self, node):
        self.map_array.append(node)

    def ekf(self):
        pass


def offline_Known_Maps():
    # dataLoc = "C:/Users/Shaul/Desktop/Msc/camera_room_4"
    map_array = pickle_functions.load_var('thesis_map.pickle')
    map_train = pickle_functions.load_var('train_map.pickle')
    for j in map_train:
        node1 = j
        Localize_with_map(node1, map_array)


def Localize_with_map(node, map_array):
    print "Start localization"
    timer = time.time()
    curr_dm_scores_sonar = np.zeros(20)
    curr_dm_scores_vision = np.zeros(20)
    curr_dm_fused = np.zeros(20)
    counter = 0
    for i in map_array:
        node2 = i
        sonar_score = node.sonar.scoring_process(node.sonar,node2.sonar,True)
        vision_score = node.vision.scoring_process(node.vision.visionPFN, node2.vision.visionPFN,False)
        fused_score = node2.bayesian_fusion(sonar_score,vision_score)
        curr_dm_scores_sonar[counter] = sonar_score
        curr_dm_scores_vision[counter] = vision_score
        curr_dm_fused[counter] = fused_score
        counter += 1
    print "Locate in %s seconds"%(time.time()-timer)


    print "Plotting"
    heat.plotToScreen(10,2,vision=curr_dm_scores_vision,sonar=curr_dm_scores_sonar, combined=curr_dm_fused)


class odom_filter:
    def __init__(self):
        process_noise_mag = 0.02
        sensor_noise_mag = 0.29
        self.state_estimate = np.array([0,0])
        self.A = np.eye(2)
        self.B = np.array([[1,0],[0,1]])
        self.C = np.array([[1,0],[0,0]])
        self.Q = sensor_noise_mag**2.0 * np.eye(2)
        self.R = process_noise_mag**2.0 * np.dot(self.B,self.B.transpose())
        self.P = self.R
        self.odom_tot = np.array([0,0])

    def update(self, node1, node2):
        odom_val = newMap.calculate_distance_by_odom(node2, bias)
        d_c = newMap.calculate_distance_by_sonar(node1, node2)
        sens = np.asarray([self.state_estimate[0] + d_c, 0])
        self.state_estimate = np.array([odom_val[0], odom_val[1]])
        self.P = self.A.dot(self.P).dot(self.A.transpose())+ self.R
        K = self.P.dot(self.C.transpose()).dot(np.linalg.inv(self.C.dot(self.P).dot(self.C.transpose())+self.Q))
        self.state_estimate = self.state_estimate + K.dot(sens.transpose()-self.C.dot(self.state_estimate.transpose()))
        self.P = (np.eye(2)-K.dot(self.C)).dot(self.P)
        self.odom_tot = np.array([odom_val[0],odom_val[1]])
        return self.state_estimate, self.odom_tot


if __name__ == '__main__':
    track_cam = tracker()
    track_cam.extract_from_file('C:/Users/Shaul/Dropbox/MultiModalSLAM/Data/Camera_room_expirments/Camera_room2/camera_track/')
    tracker = np.array([0,0])
    counter = 0
    for i in track_cam.data:
        track_cam.data[counter] = -(i - track_cam.data[0])
        counter += 1

    for i in track_cam.data:
        track_d = np.linalg.norm(i)
        track_phi = np.arctan2(-i[0],i[1])
        tracker = np.row_stack((tracker, np.asarray([track_d, track_phi])))

    heat = heatMap()
    dataLoc = 'C:/Users/Shaul/Dropbox/MultiModalSLAM/Data/Camera_room_expirments/Camera_room2/camera_room_2'
    map_array = pickle_functions.load_var('thesis_map.pickle')
    map_array.sort()
    # map_train = pickle_functions.load_var('train_map.pickle')
    newMap = Map()
    bias = np.array([map_array[0].odom.x,map_array[0].odom.y])
    Kfilter = odom_filter()
    predcitor = np.array([0,0])
    odomor = np.array([0,0])
    for i in range(1,20):
        predcit, odom = Kfilter.update(map_array[i-1],map_array[i])
        predcitor = np.row_stack((predcitor, predcit))
        odomor = np.row_stack((odomor, odom))
    ax = plt.subplot(111, projection='polar')
    ax.plot(predcitor[...,1], predcitor[...,0], '-ro', linewidth=1)
    ax.plot(odomor[...,1], odomor[...,0], '-bo', linewidth=0.5)
    ax.plot(tracker[...,1], tracker[...,0], '-go', linewidth=0.2)
    plt.show()

    # tool= Tools()
    # tool.genrate_huge_map('thesis_map.pickle')
    # dataLoc = "C:/Users/Shaul/Desktop/Msc/camera_room_4"
    # tool.genrate_huge_map('train_map.pickle')
