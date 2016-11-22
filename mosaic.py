
import os
from pylab import *


dirname = "C:/Users/Shaul/Dropbox/MultiModalSLAM/Data/Camera_room_expirments/Camera_room2/camera_room_2"

def gen_speco(dataLoc, flagIn, fileName):
    data = genfromtxt(dataLoc, delimiter=',')
    Fs = 250000
    Pxx, freqs, bins, im = specgram(data, NFFT=250, Fs=Fs, noverlap=249)
    saveInto = dirname + "/%s.png" % fileName
    print saveInto
    savefig(saveInto)
    print "did %d spec" % flagIn
    del Pxx, freqs, bins, im, saveInto, flagIn, Fs, data
    pass


def mosaic(flag):
    try:
        name = dirname
        csvFiles = {}
        for path, subdirs, files in os.walk(name):
            for fileNames in files:
                if fileNames.endswith(".csv"):
                    if (fileNames[:-4] + ".png") in files:
                        continue
                    csvFiles[fileNames[:-4]] = os.path.join(path, fileNames)
        print len(csvFiles)
        for name in csvFiles:
            gen_speco(csvFiles[name], flag, name)
            flag += 1
    except:
        mosaic(0)

if __name__ == '__main__':
    flag = 0
    mosaic(flag)