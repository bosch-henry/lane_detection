from config import *
from network.sync_batchnorm.replicate import patch_replication_callback
from network.bv_parsing_net import *
from data_process.point_transform import *
from data_process.bv_data_process import *
from util.file_util import *

import glob
import os
import torch


def Inference(model, bv_data, device):  #torch框架完成数据推理
    h, w = bv_data.shape[:2]
    input_data = np.zeros((1, 2, h, w)).astype("float32")

    input_data[0, 0, :, :] = bv_data[:, :, 0]
    input_data[0, 1, :, :] = bv_data[:, :, 1]

    #input_data = torch.from_numpy(input_data).cuda()
    input_data = torch.from_numpy(input_data).to(device)
    _, _, output_label_p_map = model(input_data)

    output_label_p_map = np.array(output_label_p_map.detach().cpu()).squeeze()

    output = np.argmax(output_label_p_map, 0)
    label_map = np.asarray(output).astype("uint8")

    return label_map


#LIDAR_IDs is defined in config.py
BV_RANGE_SETTINGS = GetBVRangeSettings(LIDAR_IDs)

if not os.path.exists(POINTS_WITH_CLASS_FOLDER):
    os.mkdir(POINTS_WITH_CLASS_FOLDER)

model = BVParsingNet()
device = ('cuda' if torch.cuda.is_available() else 'cpu')
#model = torch.nn.DataParallel(model, device_ids=GPU_IDs)
model = torch.nn.DataParallel(model)
patch_replication_callback(model)
#model = model.cuda()
checkpoint = torch.load(MODEL_NAME, map_location = 'cpu')
model = model.to(device)
model.load_state_dict(checkpoint['state_dict'])


test_data_subfolders = glob.glob(os.path.join(TEST_DATA_FOLDER, "*"))
test_data_subfolders.sort()
print(test_data_subfolders)
#vars' name end with "_set" means it containes some items which are indexed by lidar_name 
for subfolder in test_data_subfolders:
    print("processing %s" % subfolder)

    pointcloud_name_set_list, _, para_name_set = GetTestDataList(subfolder, LIDAR_IDs)
    #print(pointcloud_name_set_list)
    parameters = ReadSelectedPara(para_name_set)  #加载Lidar的外参

    output_path = os.path.join(POINTS_WITH_CLASS_FOLDER, subfolder.split("/")[-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for j, pc_name_set in enumerate(pointcloud_name_set_list):
        print("%d / %d" % (j+1, len(pointcloud_name_set_list)))
        #print(pc_name_set)

        points_input_set = ReadSelectedPoints(pc_name_set)
        #print(points_input_set)
        points_trans_set = ProjectPointsToWorld(points_input_set, parameters)
        points_merge = MergePoints(points_trans_set)

        points_near_ground = SelectPointsNearGround(points_merge, BV_COMMON_SETTINGS)
        AdjustIntensity(points_near_ground, BV_COMMON_SETTINGS)

        bv_data = ProduceBVData(points_near_ground, BV_COMMON_SETTINGS, BV_RANGE_SETTINGS)
        bv_label_map = Inference(model, bv_data, device)
        #print(bv_label_map)

        points_class_set = GetPointsClassFromBV(points_trans_set, bv_label_map, BV_COMMON_SETTINGS, BV_RANGE_SETTINGS)
        #print(points_class_set)

        OutputPointsWithClass(points_input_set, points_class_set, output_path, pc_name_set, para_name_set)

