#!/usr/bin/env python
from numpy import arange, genfromtxt, zeros, exp, amax
from scipy.signal import chirp, convolve
from scipy import fft
from peakutils import *
import matplotlib.pyplot as plt
import time
from pylab import *
from scipy.signal import butter, lfilter, freqz



Fs = 250e3
t = arange(0, 0.01, 1 / Fs)
TheoreticalChirp = chirp(t, 120e3, 0.01, 20e3)
TheoreticalChirp = TheoreticalChirp[1000:]



def smooth(x,window_len=11,window='hanning'):
    import numpy
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

#TODO: ATTENUATION




def gen_decay_by_r(convPk):
    decayed = zeros(len(convPk))
    for k in range(1, len(convPk)):
        #TODO: square to one
        r_square = ((((k / Fs) * 343.2) / 2))
        decayed[k] = convPk[k] * r_square
    return decayed





record = 's1.csv'
chain = False
doPlot = True



class SonarSignal:
    def __init__(self, record_path):
        self.raw = genfromtxt(record_path, delimiter=',')
        # self.distance, self.fft, self.numOfEchos = self.filterRAW_by_fetures(self.raw)
        self.features = self.filterRAW_by_fetures()

    def filterRAW_by_fetures(self):
        # Filter Raw file with the theoretical chirp
        if doPlot:
            Pxx, freqs, bins, im = specgram(self.raw,NFFT=250,Fs=Fs,noverlap=249)
            plt.plot()
            plt.figure()
        #
        convPk = convolve(self.raw[2800::], TheoreticalChirp[::-1])
        if doPlot:
            Pxx, freqs, bins, im1 = specgram(convPk, NFFT=250, Fs=Fs, noverlap=249)
            plt.plot()
            plt.figure()
        r = arange(0, len(convPk)).dot(343.2/(2*Fs))
        decay = convPk * r**2
        if doPlot:
            # Pxx, freqs, bins, im2 = specgram(decay, NFFT=250, Fs=Fs, noverlap=249)
            plt.plot(decay)
            plt.figure()
            plt.plot(abs(decay))
        filtered = smooth(abs(decay),window_len=50, window='hanning')
        base = baseline(filtered, 10)
        if doPlot:
            plt.plot(filtered-base)
            plt.figure()
        # Locate peaks
        signal =filtered-base
        th = 5 * mean((signal/max(signal)))
        pkslocs = indexes(signal, thres= th, min_dist=300)
        print pkslocs
        plt.plot(signal.T)
        # for loc in pkslocs:
            # point = [loc,signal[loc]]
            # plt.plot(point)




        # for loc in pkslocs:
        #     r = ((((loc / Fs) * 343.2) / 2))
        #     a = decay[loc-100:loc+100]
        #     a = abs(fft(a))
        #     plt.plot(a[:len(a) / 2])
        #     print "the radius is:", r
        # plt.show()
        return 0


class SonarFeature:
    def __init__(self, distance, fft_describe):
        self.distance = distance
        self.fft = fft_describe




if __name__ == '__main__':
    # experiment_location = sys.argv[1]
    if chain:
        for k in range(0, 10):
            record = ('s%s.csv' % k)
            rec1 = 'C:/Users/Shaul/Desktop/Msc/Thesis/Expirment/alone10.4/alone_5/'+record
            # rec2 = 'C:/Users/Shaul/Desktop/Msc/Thesis/Expirment/alone10.4/alone_5/s4.csv'
            sonar_feature1 = SonarSignal(rec1)
            # sonar_feature2 = SonarSignal(rec2)
    else:
        rec1 = 'C:/Users/Shaul/Dropbox/MultiModalSLAM/Data/alone_5/s134.csv'
        # rec2 = 'C:/Users/Shaul/Desktop/Msc/Thesis/Expirment/alone10.4/alone_5/s4.csv'
        sonar_feature1 = SonarSignal(rec1)


