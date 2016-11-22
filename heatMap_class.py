import os
import numpy as np
import matplotlib.pyplot as plt



__author__ = 'Shaul'

class heatMap:
    def __init__(self):
        fo = os.open('log.txt', os.O_RDWR | os.O_CREAT)
        self.fd = os.fdopen(fo, 'w+')
        self.sonar = None
        self.vision = None
        self.temp = None
        self.track = None
        self.combines = None
        self.combines_plot = None
        self.first=True


    def arrange(self, num_of_DM_per_rows, num_of_column,combined = None ,sonar = None, vision = None, track=None, camera = None):
        if combined is not None:
            comined_arranged = np.asarray(combined)
            self.combines = self.arrData(comined_arranged, num_of_DM_per_rows, num_of_column)
        if sonar is not None:
            sonar_arranged = np.asarray(sonar)
            self.sonar = self.arrData(sonar_arranged, num_of_DM_per_rows, num_of_column)
        if vision is not None:
            vision_arranged = np.asarray(vision)
            self.vision = self.arrData(vision_arranged, num_of_DM_per_rows, num_of_column)
        # temp_arranged = np.asarray(toPlot['temp'])
        # self.temp = arrData(temp_arranged, num_of_DM_per_rows, num_of_column)
        if track is not None:
            self.track = np.asarray(track)
        if camera is not None:
            self.camera = np.asarray(camera)

    def plotToScreen(self, num_of_DM_per_rows, num_of_column, combined=None, sonar=None, vision=None, track=None, camera=None, single=False):
        if not self.first:
            self.update_plots(num_of_DM_per_rows, num_of_column, combined=combined, sonar=sonar, vision=vision, track=track, camera=camera)
        self.first=False
        self.arrange(num_of_DM_per_rows, num_of_column, combined=combined, sonar=sonar, vision=vision, track=track, camera=camera)
        plt.subplot(1,4,1)
        if self.combines is not None:
            plt.title('Combined')
            himage = plt.imshow(self.combines.transpose(), interpolation='bilinear',cmap='YlOrRd',vmin=self.combines.T.min(),vmax=self.combines.T.max())
        # plt.pcolor(self.combines.transpose(), cmap='Blues')
        # plt.contourf(np.arange(0,4,1), np.arange(0,11,1), self.combines.transpose(), 100)
        # plt.clim(0, 1)
        # plt.axis([0, num_of_column,0, num_of_DM_per_rows])
        plt.subplot(1,4,2)
        if self.sonar is not None:
            plt.title('Sonar')
            plt.imshow(self.sonar.transpose(), interpolation='nearest',cmap='YlOrRd',vmin=0,vmax=1)
        # plt.pcolor(self.sonar.transpose(), cmap='Blues')
        # plt.clim(0, 1)
        # plt.axis([0, num_of_column,0, num_of_DM_per_rows])
        plt.subplot(1,4,3)
        if self.vision is not None:
            plt.title('Vision')
            plt.imshow(self.vision.transpose(), interpolation='nearest',cmap='YlOrRd',vmin=0,vmax=1)
        # plt.pcolor(self.vision.transpose(), cmap='Blues')
        # plt.clim(0, 1)
        # plt.axis([0, num_of_column,0, num_of_DM_per_rows])
        # plt.colorbar()
        plt.subplot(1,4,4)
        if self.track is not None:
            plt.title('Track')
            plt.plot(-self.track[...,1], -self.track[...,0].T,'-ro', -self.camera[..., 0], self.camera[..., 1].T,'-go')
        # self.fd.write(str(self.camera[..., 1].T) + ' ' + str(-self.camera[..., 0]) + '\n')
        # plt.axis([-0.3, 3, 0, 3])
        mng = plt.get_current_fig_manager()
        mng.window.state('zoomed')
        # print himage._A
        plt.show(block=single)

    def update_plots(self, num_of_DM_per_rows, num_of_column, combined=None, sonar=None, vision=None, track=None, camera=None, single=False):
        self.arrange(num_of_DM_per_rows, num_of_column, combined=combined, sonar=sonar, vision=vision, track=track, camera=camera)
        plt.subplot(1,4,1)
        if self.combines is not None:
            himage = plt.imshow(self.combines.transpose(), interpolation='bilinear',cmap='YlOrRd',vmin=self.combines.T.min(), vmax=self.combines.T.max())
        # plt.pcolor(self.combines.transpose(), cmap='Blues')
        # plt.clim(0, 1)
        plt.subplot(1,4,2)
        if self.sonar is not None:
            plt.imshow(self.sonar.transpose(), interpolation='nearest',cmap='YlOrRd',vmin=0,vmax=1)
        # plt.pcolor(self.sonar.transpose(), cmap='Blues')
        # plt.clim(0, 1)
        plt.subplot(1,4,3)
        if self.vision is not None:
            plt.imshow(self.vision.transpose(), interpolation='nearest',cmap='YlOrRd',vmin=0,vmax=1)
        # plt.pcolor(self.vision.transpose(), cmap='Blues')
        # plt.clim(0, 1)
        plt.subplot(1,4,4)
        if self.track is not None:
            plt.plot(-self.track[...,1], -self.track[...,0].T,'-ro', -self.camera[..., 0], self.camera[..., 1].T,'-go')
            self.fd.write(str(-self.track[...,1]) + ' ' + str(-self.track[...,0]) + '\n')
        # plt.clim(0, 1)
        plt.draw()
        # print himage._A

        plt.pause(0.1)

    def update_data(self):
        self.combines_plot.set_data(self.combines.transpose())
        return self.combines_plot


    def arrData(self, data, num_of_DM_per_rows, num_of_column):
        for multiplier in range(1, num_of_column, 2):
            backward_data = data[num_of_DM_per_rows * multiplier : num_of_DM_per_rows * multiplier + num_of_DM_per_rows]
            backward_data = backward_data[::-1]
            data[num_of_DM_per_rows * multiplier : num_of_DM_per_rows * multiplier + num_of_DM_per_rows] = backward_data
            test = data.reshape(num_of_column, num_of_DM_per_rows)
        return test
