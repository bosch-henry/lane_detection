
# Selected lidar ids
LIDAR_IDs = ["1","2","3","4","5","6"]

# Selected line side ids
LINE_SIDE_IDs = ["left", "right"]

# Bird view map setttings
BV_COMMON_SETTINGS = { 
                # Height shift to make the z-axis value of ground be 0
                "train_height_shift": 0,
                # Minimum z-axis value of the interval to select points near the ground
                "shifted_min_height": -1,
                # Maximum z-axis value of the interval to select points near the ground
                "shifted_max_height": 1,
                # 1 meter in x-axis corresponds to "distance_resolution_train" pixels on the bird view map
                "distance_resolution_train": 6,
                # 1 meter in y-axis corresponds to "width_resolution_train" pixels on the bird view map
                "width_resolution_train": 30,
                # Point radius on the bird view  俯视野下点云半径
                "point_radius_train": 1.5,
                # If intensity value of one point is bigger thatn truncation_max_intensiy, the intensity will be set to truncation_max_intensiy
                # 低通滤波，所有反射强度高于0.12的点将强度设置为0.12
                "truncation_max_intensiy": 0.12,
                # Intensity shift to make the area with points(intensity may be 0) different with that without points （区分有点区域与无点区域）
                "train_background_intensity_shift": 0.1
                }

# Line plot settings
LINE_PLOT_SETTINGS = {
                "left" : {"color" : "red", "label" : "left line"},
                "right" : {"color" : "green", "label" : "right line"}
                }

MODEL_NAME = "./model/livox_lane_det.pth"
GPU_IDs = [0]

TEST_DATA_FOLDER = "./test_data"
POINTS_WITH_CLASS_FOLDER = "./result/points_with_class/"
VIS_FOLDER = "./result/points_vis/"
LINE_SCATTER_FOLDER = "./result/line_scatter_matrix"
LINE_FITTING_FOLDER = "./result/line_fitting"


# Get the bird view range accrording to the selected lidar ids
def GetBVRangeSettings(lidar_ids):
    # Default settings
    bv_settings = { 
                    # farthest detection distnace in front of the car
                    "max_distance": 100,
                    # farthest detection distnace behind the car
                    "min_distance": -40,
                    # farthest detection distance to the left of the car
                    "left_distance": 20,
                    # farthest detection distance to the right of the car
                    "right_distance": 20}
    
    assert(len(lidar_ids) > 0)
    for lidar_id in lidar_ids:
        assert(int(lidar_id) in range(1, 7))

    if "6" in lidar_ids:
        bv_settings["max_distance"] = 100
    elif "1" in lidar_ids or "2" in lidar_ids or "5" in lidar_ids:
        bv_settings["max_distance"] = 40
    else:
        bv_settings["max_distance"] = 0

    if "2" not in lidar_ids and "3" not in lidar_ids and "1" not in lidar_ids and "6" not in lidar_ids:
        bv_settings["left_distance"] = 0

    if "4" not in lidar_ids and "5" not in lidar_ids and "1" not in lidar_ids and "6" not in lidar_ids:
        bv_settings["right_distance"] = 0

    if "3" not in lidar_ids and "4" not in lidar_ids:
        bv_settings["min_distance"] = 0

    return bv_settings


# Get the line fitting config 
def GetLineFittingSetting(method="OLS", degree=2):
    if method == "OLS":
        OLS_setting = {
            # define the regression degree for OLS method
            "OLS_degree" : degree - 1
        }
        return OLS_setting
    elif method == "RANSAC":
        RANSAC_setting = {
            # define the regression degree for RANSAC
            "RANSAC_degree" : degree,
            # define the maxmum number of iterations for random sample selection
            "RANSAC_max_trials" : 100,
            # define the minimum number of samples chosen randomly from original data
            "RANSAC_min_samples" : 50,
            # define the loss method for inner point selection
            "RANSAC_loss" : 'absolute_loss',
            # define the threshold alowed for loss float
            "RANSAC_residual_threshold" : None,
            # define the threshold for iteration stop probability float in range [0, 1]
            "RANSAC_stop_probability" : 0.99
        }
        return RANSAC_setting
    else:
        return None
    