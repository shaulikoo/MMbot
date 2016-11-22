from simplejson import loads
from sonar_process import SonarSignal
from Image_Visual_Features import ImageVisualFeatures
from cv2 import imread
from time import time


class ScanData:
    def __init__(self, pathToJson, experiment_location):
        jsonfile = loads(open(pathToJson).read())
        self.capture = {}
        for data in jsonfile:
            self.capture[data['angle']] = CaptureData(data['capture'], experiment_location)
        self.odom_x = jsonfile[1]['capture']['odom_x']
        self.odom_y = jsonfile[1]['capture']['odom_y']
        self.phi = jsonfile[1]['capture']['phi']
        self.numOfCaptures = len(self.capture)
        self.meta = pathToJson
        self.time = jsonfile[1]['capture']['time']


class CaptureData:
    def __init__(self, data, experiment_location):
        self.vision = self.get_features_from_rgb_image(experiment_location + data['Image'])
        self.sonar = SonarSignal(experiment_location + data['Sonar'])
        self.temp = data['Temperture']

    def get_features_from_rgb_image(self, rgb_image_path):
        # start_time = time.time()
        bgr_image = imread(rgb_image_path)
        visual_features = ImageVisualFeatures("SIFT", bgr_image)
        # print (" ***get features from image: " + rgb_image_path + ' time: ' + str(time.time() - start_time) + "***\n")
        return visual_features
