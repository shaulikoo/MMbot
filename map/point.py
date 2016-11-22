import numpy as np
from scipy.interpolate import interp1d


class MapPoint():
    def __init__(self):
        self.x = 0
        self.y = 0
        self.data = np.zeros(100)


class ScanPoint():
    def __init__(self):
        self.resulotion = 200000
        self.data = np.zeros(100)
        self.m = interp1d([0, self.resulotion],[1,0])

    def creGaussian(self, mapPoints, X, Y):
        errCalc = 0
        for summerize in range(0, len(mapPoints.data)):
            errCalc += (self.data[summerize] - mapPoints.data[summerize])**2
        if errCalc > self.resulotion : errCalc = self.resulotion
        errCalc = float(self.m(errCalc))
        Z = errCalc * np.exp(-(((X-mapPoints.x)**2)/((2*0.6)**2)+((Y-mapPoints.y)**2))/((2*0.5)**2))
        return Z
