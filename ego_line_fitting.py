from config import *
import glob
import os
from data_process.line_fitting_core import *
from util.file_util import *
import matplotlib.pyplot as plt

# make path for line fitting result
if not os.path.exists(LINE_FITTING_FOLDER):
    os.mkdir(LINE_FITTING_FOLDER)

# set list for original data subfolders
original_data_subfolders = glob.glob(os.path.join(LINE_SCATTER_FOLDER, "*"))
original_data_subfolders.sort()

# process for each original data subfolder
for subfolder in original_data_subfolders:
    print("processing %s" % subfolder)
    # get the line scatter file name list
    line_scatter_name_set_list = GetLineScatterDataList(subfolder, LINE_SIDE_IDs)
    # make output path for each subfolder input
    output_path = os.path.join(LINE_FITTING_FOLDER, subfolder.split("/")[-1])
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    # linear regression for each 100ms scatter points of all line sides
    for j, line_side_set in enumerate(line_scatter_name_set_list):
        print("%d / %d" % (j+1, len(line_scatter_name_set_list)))
        #print(line_side_set)

        # get one scatter set for each line_side
        scatter_input_set = ReadLineScatterPoints(line_side_set)
        #print(scatter_input_set)
        # init prediction scatter dict and coef dict for each line_side
        line_scatter_pred_matrix = {}
        line_coef = {}

        # line fitting for each line_side input scatter via line fitting core and fill result into pred_matrix and coef_list
        for line_side in line_side_set.keys():
            line = LineFitting(scatter_input_set[line_side], method="OLS", degree=3)
            line.line_regression()
            line_scatter_pred_matrix[line_side] = line.get_regression_matrix()
            line_coef[line_side] = line.get_coef()
            line.line_plotting(LINE_PLOT_SETTINGS[line_side]["color"], LINE_PLOT_SETTINGS[line_side]["label"])

        # write result into specific files (path structure same as input scatters)
        OutputLineFittingResult(line_scatter_pred_matrix, line_coef, line_side_set, output_path)

        # ploting line scatter and regression result
        plt.legend(loc='lower right')
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        #plt.savefig("./plot.jpg", dpi=300)
        plt.show()