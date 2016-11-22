__author__ = 'Shaul'
from numpy import genfromtxt,column_stack

class tracker:
    def __init__(self):
        self.data = None

    def extract_from_file(self, tracker_path):
        tracker_path_x = tracker_path + 'x_axis.csv'
        tracker_path_y = tracker_path + 'y_axis.csv'
        x = genfromtxt(tracker_path_x, delimiter=',')
        y = genfromtxt(tracker_path_y, delimiter=',')
        self.data = column_stack((x, y))


