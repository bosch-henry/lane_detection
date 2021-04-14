from config import *
from util.color_table_for_class import color_table_for_class
from util.file_util import *
from data_process.point_transform import *
from data_process.bv_data_process import *

import os
import glob

#LIDAR_IDs is defined in config.py
BV_RANGE_SETTINGS = GetBVRangeSettings(LIDAR_IDs)

if not os.path.exists(VIS_FOLDER):
    os.mkdir(VIS_FOLDER)

test_data_subfolders = glob.glob(os.path.join(POINTS_WITH_CLASS_FOLDER, "*"))
#print(test_data_subfolders)
test_data_subfolders.sort()

#vars' name end with "_set" means it containes some items which are indexed by lidar_name 
for subfolder in test_data_subfolders:
    print("processing %s" % subfolder)
    subfolder = os.path.join(subfolder, "id00000_data4_2020_05_11")

    pointcloud_name_set_list, _, para_name_set = GetTestDataList(subfolder, LIDAR_IDs)

    parameters = ReadSelectedPara(para_name_set)

    output_path = os.path.join(VIS_FOLDER, subfolder.split("/")[-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for j, pc_name_set in enumerate(pointcloud_name_set_list):
        print("%d / %d" % (j+1, len(pointcloud_name_set_list)))
        #print(pc_name_set)

        points_input_set = ReadSelectedPoints(pc_name_set)
        points_trans_set = ProjectPointsToWorld(points_input_set, parameters)
        #print(points_trans_set)
        points_merge = MergePoints(points_trans_set)
        AdjustIntensity(points_merge, BV_COMMON_SETTINGS)

        vis_img = VisualizePointsClass(points_merge)
        cv2.imshow("1",vis_img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        SaveVisImg(vis_img, pc_name_set, output_path)
        #print(output_path)