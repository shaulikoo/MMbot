from time import time
import os
from pickle_functions import load_var, save_var
import score_functions
from read_json import ScanData
import numpy as np
import matplotlib.pyplot as plt
from DataMatrixClasses import DataMatrix
from score_functions import bayes_fusion,Kalman_filter
from extract_tracker import tracker
from scipy import interpolate
from heatMap_class import heatMap

num_of_DM_per_rows = 11
num_of_column = 4
generate_map = False
for_one = False
gen_dm = False
single_val = 3


# save_name = 'maps_pickles/map_5_dec.pickle'
save_name = 'maps_pickles/camera_room_1.pickle'
# dir = 'C:/Users/Shaul/Dropbox/MultiModalSLAM/Data/Camera_room_expirments/Camera_room1/bagfiles/camera_room_1'
dir = 'C:/Users/Shaul/Dropbox/MultiModalSLAM/Data/Camera_room_expirments/Camera_room2/camera_room_2'
# dir = 'C:/Users/Shaul/Desktop/Msc/camera_room_4'
# dir = 'C:/Users/Shaul/Desktop/Msc/Thesis/Expirment/alone10.4/alone_3'

# nndist_val = 0.9; fft_weight_val = 0.5; sigma_fft_val = 0.05; sigma_distance_val = 0.1
nndist_val = 0.9; fft_weight_val = 0.5; sigma_fft_val = 80; sigma_distance_val = 0.1

#TODO: Normalize by signal transmit

def main(experiment_location):
    heat = heatMap()
    total_time_start = time()
    # ######################  get raw data  ##########################
    if generate_map:
        print "Start generate map ..."
        jsonfiles = []
        for path, subdirs, files in os.walk(experiment_location):
            for fileNames in files:
                if fileNames.endswith(".json"):
                    jsonfiles.append(os.path.join(path, fileNames))
        # ###################### create DataMatrix ######################
        if len(jsonfiles) == 0:
            print "Error - No Json files"
        else:
            print "No. of Json file: %s" % len(jsonfiles)
            map = []
            for paths in jsonfiles:
                start_time = time()
                map.append(DataMatrix(ScanData(paths, experiment_location)))
                print "Finish %s DataMatrixes" % len(map)
                print "time: " + str(time() - start_time)
            map.sort()
            save_var(map, save_name)
            exit()
    else:
        print "Loading Map from memory ..."
        map = load_var(save_name)
        print "Map on memory"



    init=1
    if for_one:
            single_DM = '/scan%s.json'%single_val
            if gen_dm:
                # print "Start Check DM"
                checkPoint = DataMatrix(ScanData(experiment_location+single_DM, experiment_location))
                save_var(checkPoint, 'DM.pickle')
                # print "End Check DM"
            else:
                checkPoint = load_var('DM.pickle')
            scores_vision, scores_sonar = score_functions.calc_1_vs_all_scores_by_feature(map, checkPoint, 'all')
            bayes = bayes_fusion(len(map))
            scores = score_functions.fusion(scores_vision, scores_sonar, bayes)
            scores_only_sonar = scores_sonar
            scores_only_vision = scores_vision
            # scores_only_temp = score_temp
            heat.arrange(num_of_DM_per_rows, num_of_column, combined = scores, sonar = scores_only_sonar, vision = scores_only_vision)
            heat.plotToScreen(True)
            # print "End Fusions scores"
            # pltHeatMaps(scores, scores_only_sonar, scores_only_vision, init)

    else:
        try:
            track_cam = tracker()
            track_cam.extract_from_file('C:/Users/Shaul/Dropbox/MultiModalSLAM/Data/Camera_room_expirments/Camera_room2/camera_track/')
            bayes = bayes_fusion(len(map))
            track = None
            for json_num in range(1, 25):
                checkPoint = DataMatrix(ScanData(dir + '/scan%s.json' % json_num, dir))
                point = np.array([checkPoint.odom_x, checkPoint.odom_y, checkPoint.phi])
                if track is None:
                    bias = point
                    track = np.array(point-bias)
                else:
                    track = np.row_stack((track, point-bias))
                scores_vision, scores_sonar= score_functions.calc_1_vs_all_scores_by_feature(map, checkPoint, 'all')
                scores = score_functions.fusion(scores_vision, scores_sonar, bayes)
                # heat.arrange(num_of_DM_per_rows, num_of_column,combined=scores, sonar=scores_sonar, vision=scores_vision, track=track, camera=track_cam.data)
                if json_num == 1:
                    heat.plotToScreen(num_of_DM_per_rows, num_of_column,combined=scores, sonar=scores_sonar, vision=scores_vision, track=track, camera=track_cam.data)
                else:
                    heat.update_plots(num_of_DM_per_rows, num_of_column,combined=scores, sonar=scores_sonar, vision=scores_vision, track=track, camera=track_cam.data)
                # pltHeatMaps(scores, scores_only_sonar, scores_only_vision, init)
                print "End %s Plot" % json_num
                init = 0

        except KeyboardInterrupt:
            print "Keyboard Interrupt"
            heat.fd.close()

        except SystemExit:
            print "SystemExit Interrupt"
            heat.fd.close()

        finally:
            print "Final"
            heat.fd.close()
    print "Total time: %s" % str(time() - total_time_start)


def print_odom(checkpoint,map, max):
    point_map = np.array([map[max[0]].odom_x-map[0].odom_x, map[max[0]].odom_y-map[0].odom_y, map[max[0]].phi-map[0].phi])
    print "The Odometry on map is: %s" % point_map
    print "The Odometry on dm is: %s" % checkpoint

if __name__ == '__main__':
    main(dir)