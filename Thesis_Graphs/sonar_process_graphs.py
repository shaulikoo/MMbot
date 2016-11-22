__author__ = 'Shaul'

import numpy as np
from scipy.signal import chirp, convolve,find_peaks_cwt
from scipy import fft, stats
import pylab as plt
import peakutils
import peakutils.plot


dir = "C:/Users/Shaul/Google Drive/Thesis/figures/sonar/process/rons/"

doPlot = False
doPlot2 = False
doPlot3 = False
fig_num=1

Fs = 250e3
Cs = 343.42
C = 5.0

t = np.arange(0, 0.01, 1 / Fs)
TheoreticalChirp = chirp(t, 120e3, 0.01, 20e3)
TheoreticalChirp = TheoreticalChirp[1000:]
if doPlot2:
    plt.figure(fig_num)
    Pxx, freqs, bins, im = plt.specgram(TheoreticalChirp[::-1],NFFT=250,Fs=Fs,noverlap=249)
    plt.plot()
    fig_num+=1
    plt.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.savefig(dir + 'theortical_chirp.tiff')



def smooth(x,window_len=11,window='hanning'):
    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s = np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

#Input
rec1 = "C:/Users/Shaul/Dropbox/MultiModalSLAM/Data/RSL/ego_motion5/s0.csv"
# rec1 = 'C:/Users/Shaul/Dropbox/MultiModalSLAM/Data/alone_5/s146.csv'
raw = np.genfromtxt(rec1, delimiter=',')

if doPlot:
    plt.figure(fig_num)
    Pxx, freqs, bins, im = plt.specgram(raw,NFFT=250,Fs=Fs,noverlap=249)
    plt.plot()
    plt.ticklabel_format(axis='y',style='sci',scilimits=(-3,3))
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.savefig(dir + 'raw_spectogram.tiff')
    fig_num+=1
    plt.figure(fig_num)
    fig_num+=1
    plt.plot(raw)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
    plt.savefig(dir + 'raw_data.tiff')




#Match filter
convPk = convolve(raw, TheoreticalChirp[::-1])
convPk = convPk[2600:]
if doPlot3:
    plt.figure(fig_num)
    Pxx, freqs, bins, im1 = plt.specgram(convPk, NFFT=250, Fs=Fs, noverlap=249)
    plt.plot()
    plt.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.savefig(dir + 'match_filter_spec.tiff')
    fig_num+=1
    a=False
    if a:
        plt.figure(fig_num)
        fig_num+=1
        plt.plot(convPk)
        plt.tick_params(axis='both', which='major', labelsize=17)
        plt.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
        plt.savefig(dir + 'match_filter_saptial.tiff')



#Fix Decay
r = np.arange(0, len(convPk)).dot(Cs/(2.0*Fs))
# decay = convPk * r ** 2.0
decay = convPk * r


if doPlot2:
    plt.figure(fig_num)
    plt.plot(decay)
    fig_num+=1
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
    plt.savefig(dir + 'match_filter_decay_saptial.tiff')


filtered = smooth(decay,window_len=11, window='hanning')
if doPlot3:
    plt.figure(fig_num)
    plt.plot(filtered)
    fig_num+=1
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
    plt.savefig(dir + 'smooth.tiff')


base = peakutils.baseline(abs(filtered), 10)
signal =abs(filtered)-base
if doPlot2:
    plt.figure(fig_num)
    plt.plot(signal)
    fig_num+=1
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
    plt.savefig(dir + 'smooth_sub_baseline.tiff')
if doPlot2:
    plt.figure(fig_num)
    baseline = plt.plot(base,'-g',label="Baseline")
    fig_num+=1
    data = plt.plot(filtered,'-r',label="Data")
    data = plt.plot(signal,'-b',label="$\hat f $")
    plt.legend()
    plt.savefig(dir + 'baseline_example.tiff')

signal_x = np.arange(0,len(signal))
th = C * np.mean((signal/max(signal)))
pkslocs = peakutils.indexes(signal, thres=th, min_dist=400)
if doPlot3:
    plt.figure(fig_num)
    fig_num+=1
    peakutils.plot.plot(signal_x, signal, pkslocs)
    plt.tick_params(axis='both', which='major', labelsize=17)
    plt.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
    plt.savefig(dir + 'peak_detect.tiff')




features = []
flag = 1
for loc in pkslocs:
    if loc < 100:
        continue
    a = abs(fft(decay[loc - 60:loc + 60]))
    if doPlot2:
        xf = np.linspace(0.0, 1.0/(2.0*(1.0/Fs)), len(a)/2)
        plt.figure(fig_num)
        plt.plot(xf, a[:len(a)/2])
        plt.title('Radius: %s [m]' % r[loc] ,fontsize=20)
        plt.ticklabel_format(axis='x',style='sci',scilimits=(-3,3))
        plt.tick_params(axis='both', which='major', labelsize=17)
        fig_num+=1
        plt.savefig(dir + 'fft%s.tiff'%flag)
        flag+=1

        # plt.figure(fig_num)
        # fig_num+=1
        # Pxx, freqs, bins, im1 = plt.specgram(decay[loc - 60:loc + 60], NFFT=250, Fs=Fs, noverlap=249)
        # plt.plot()


if doPlot or doPlot2 or doPlot3:
    plt.show()