#!/usr/bin/env python
from numpy import arange, genfromtxt, zeros, mean, float32, sum, row_stack, array, exp
from scipy.signal import chirp, convolve
from scipy import fft
from peakutils import indexes, baseline
import math


# Theortical chrip generator
Cs = 343.42
Fs = 250e3
C = 5.0

t = arange(0, 0.01, 1 / Fs)
TheoreticalChirp = chirp(t, 120e3, 0.01, 20e3)
TheoreticalChirp = TheoreticalChirp[1000:]


#Filter

def smooth(x, window_len=11, window='hanning'):
    import numpy

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."
    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
    s = numpy.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat':  #moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')
    y = numpy.convolve(w / w.sum(), s, mode='valid')
    return y


#TODO: ATTENUATION

class SonarSignal:
    def __init__(self, record_path):
        # Input data from the csv file location
        self.raw_record = genfromtxt(record_path, delimiter=',')
        self.features = self.filterRAW_by_fetures() #Starts the process


    def filterRAW_by_fetures(self):
        # Filter Raw file with the theoretical chirp - Match filter
        convPk = convolve(self.raw_record, TheoreticalChirp[::-1])
        convPk = convPk[2500:] # Crop the burst chirp
        r = arange(0, len(convPk)).dot(Cs/(2.0*Fs)) #Calculate radius per sample number
        decay = convPk * r# inverse-square law
        # Locate peaks
        filtered = smooth(decay, window_len=11, window='hanning') #LPF
        base = baseline(abs(filtered), 10) #baseline fitting (bias)
        signal = abs(filtered) - base #The normal data
        th = C * mean((signal / max(signal))) # dynamic threshold
        pkslocs = indexes(signal, thres=th, min_dist=300) # Peak detection
        features = []
        flag = 0
        # Genrate features
        for loc in pkslocs:
            if loc < 100:
                continue
            r_for_peak = r[loc] # The raduis
            a = abs(fft(decay[loc - 60:loc + 60])).astype(float32) # The FFT computation with 120 samples window
            features.append(SonarFeature(r_for_peak, a[:len(a) / 2])) # Packing
            flag += 1
        return features


class SonarFeature:
    def __init__(self, distance=0, fft_describe=zeros((1,60))):
        self.distance = []
        self.distance.append(distance)
        self.fft = fft_describe.reshape(1, 60)
        pass

    def concatenate_features(self, sonar_2_concatenate):
        if sonar_2_concatenate.fft is None:
            return
        self.fft = row_stack((self.fft, sonar_2_concatenate.fft))
        self.distance += sonar_2_concatenate.distance
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
            s1 = self.distance[m.queryIdx]
            s2 = feature_to_compare.distance[m.trainIdx]
            ratio = exp(-0.5*((s1-s2)/0.3)**2)
            all_scale_ratio.append(ratio)
            # if s1 > s2:
            #     ratio = exp(-0.5*((s1-s2)/0.5)**2)
            #     # all_scale_ratio.append(s2 / s1)
            #     all_scale_ratio.append(ratio)
            # else:
            #
            #     # all_scale_ratio.append(s1 / s2)
            self_matched_features_idx.append(m.queryIdx)
        # descriptor distances
        norm_mean_descriptor_dist = sum(array(all_distances)) / (len(all_distances) * descriptor_match_normalize_factor)
        # scale ratio distance
        norm_scale_ratio_distance = 1.0 - sum(array(all_scale_ratio)) / (len(all_scale_ratio))
        # num of matches distance
        norm_num_matches_dist = 1.0 - \
                                ((float(len(matches)) / float(len(self.fft)) + float(len(matches)) / float(len(feature_to_compare.fft)))/2.0)
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
