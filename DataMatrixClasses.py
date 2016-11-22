__author__ = 'Shaul'
import re


class DataMatrix:
    def __init__(self, scan_json):
        self.odom_x = scan_json.odom_x
        self.odom_y = scan_json.odom_y
        self.phi = scan_json.phi
        self.time = scan_json.time
        self.meta = scan_json.meta
        self.scan_number = int(re.findall('scan(.*?).json', self.meta)[0])
        self.scan_descriptors = scan_json.capture

    def __lt__(self, other):
        return self.scan_number < other.scan_number