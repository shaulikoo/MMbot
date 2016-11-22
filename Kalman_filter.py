import numpy as np
from pickle_functions import load_var
import matplotlib.pyplot as plt
from sonar_process import SonarFeature
import cv2


L = 0.11
I = 1. / 12. * L ** 2




class odom_kalman:
    def __init__(self):
        self.process_noise_mag = 0.02
        self.sensor_noise_mag = 0.1
        self.R = None
        self.Q = None
        self.K = None
        self.d_trans = None
        self.d_rot1 = None
        self.d_rot2 = None
        self.update_r()
        self.P = self.R
        self.H = np.eye(3)
        self.h_s = None
        self.G = None
        self.g_s = np.eye(3, dtype='float64')
        self.state_estimate = np.array([0, 0, 0])
        self.predic_state = [self.state_estimate]
        self.set_q()
        self.update_r()


    def predict(self, odom_val):
        self.d_trans = np.sqrt((odom_val[0]-self.state_estimate[0])**2+(odom_val[1]-self.state_estimate[1])**2)
        self.d_rot1 = np.arctan2((odom_val[1]-self.state_estimate[1]), (odom_val[0]-self.state_estimate[0])) - self.state_estimate[2]
        self.d_rot2 = odom_val[2]-self.state_estimate[2]-self.d_rot1
        self.g_s = np.array([self.state_estimate[0]+self.d_trans*np.cos(self.state_estimate[2]+self.d_rot1),self.state_estimate[1]+self.d_trans*np.sin(self.state_estimate[2]+self.d_rot1), self.state_estimate[2]+self.d_rot1+self.d_rot2])
        self.G = np.array([[1, 0, -self.d_trans*np.sin(self.state_estimate[2]+self.d_rot1)], [0, 1, -self.d_trans*np.cos(self.state_estimate[2]+self.d_rot2)], [0, 0, 1]])
        self.update_r()
        self.P = self.G.dot(self.P.dot(self.G.T)) + self.R

    def update_r(self):
        self.R = self.process_noise_mag ** 2*np.eye(3)

    def set_q(self):
        self.Q = self.sensor_noise_mag ** 2 + np.eye(3)

    def perception(self):
        self.h_s = np.array([self.d_trans*np.cos(self.state_estimate[2]+self.d_rot1)-self.state_estimate[0],self.d_trans*np.sin(self.state_estimate[2]+self.d_rot1)-self.state_estimate[1], self.d_rot1+self.d_rot2-self.state_estimate[2]])
        self.H = np.array([[-1, 0, -self.d_trans*np.sin(self.state_estimate[2]+self.d_rot1)], [0, -1, -self.d_trans*np.cos(self.state_estimate[2]+self.d_rot2)], [0, 0, -1]])

    def set(self, odom):
        self.update_r()

    def run_filter(self, odom, sens):
        self.predict(odom)
        self.perception()
        self.state_estimate = self.g_s
        self.K = self.P.dot(self.H.T).dot(np.linalg.inv(self.H.dot(self.P.dot(self.H.T))+self.Q))
        self.state_estimate = self.state_estimate + self.K.dot((sens - self.h_s))
        self.predic_state.append(self.state_estimate)
        self.P = (np.eye(3) - self.K.dot(self.H)).dot(self.P)


def knn_matcher(descriptors1, descriptors2, nndist=0.75):
    bf = cv2.BFMatcher()
    # Match descriptors with 2 nearest neighbours
    matches = bf.knnMatch(descriptors1.fft, descriptors2.fft, k=2)
    m = matches[0][0]
    return m


def forward_sens(DM_now, DM_prev):
    odom_now = np.zeros(3)
    odom_now[0] = DM_now.odom_x
    odom_now[1] = DM_now.odom_y
    odom_now[2] = DM_now.phi
    odom_prev = np.zeros(3)
    odom_prev[0] = DM_prev.odom_x
    odom_prev[1] = DM_prev.odom_y
    odom_prev[2] = DM_prev.phi
    scan_descriptors_1 = DM_now.scan_descriptors
    scan_descriptors_2 = DM_prev.scan_descriptors
    first_ref = True
    first_check = True
    for fet in scan_descriptors_1[0].sonar.features:
        if first_ref:
            concatenate_feature_1 = SonarFeature(fet.distance[0], fet.fft)
            first_ref = False
        else:
            concatenate_feature_1.concatenate_features(fet)
    for fet in scan_descriptors_2[0].sonar.features:
        if first_check:
            concatenate_feature_2 = SonarFeature(fet.distance[0], fet.fft)
            first_check = False
        else:
            concatenate_feature_2.concatenate_features(fet)
    matches = knn_matcher(concatenate_feature_1, concatenate_feature_2, nndist=0.75)
    s1 = concatenate_feature_1.distance[matches.queryIdx]
    s2 = concatenate_feature_2.distance[matches.trainIdx]
    if s1 > s2:
        sens = s1-s2
    else:
        sens = s2-s1
    return sens, odom_now, odom_prev


if __name__ == '__main__':
    print "Loading Map"
    map = load_var('map_5_dec.pickle')
    print "Generate sets"
    odom_save = np.array([0, 0, 0])
    prediction_save = np.array([0, 0, 0])
    Kfilter = odom_kalman()

    for i in range(2,10):
        sens, odom_now, odom_prev = forward_sens(map[i], map[i-1])
        odom_save = np.row_stack((odom_save, odom_now))
        Kfilter.run_filter(odom_now, sens)
        prediction_save = np.row_stack((prediction_save,Kfilter.state_estimate))

    Kfilter1 = odom_kalman()
    Kfilter1.state_estimate = Kfilter.state_estimate
    Kfilter1.predic_state = [Kfilter1.state_estimate]
    for i in range(10,18):
        sens, odom_now, odom_prev = forward_sens(map[i], map[i-1])
        odom_save = np.row_stack((odom_save, odom_now))
        Kfilter1.run_filter(odom_now, sens)
        prediction_save = np.row_stack((prediction_save, Kfilter1.state_estimate))



    print odom_save[...,0]
    plt.plot(odom_save[...,0],odom_save[...,1],'-ro')
    plt.plot(prediction_save[...,0],prediction_save[...,1],'-bo')

    plt.show()












