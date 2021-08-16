import numpy as np
import os


path = "sensitivity_margin/"

# # file = "result_isoforest.out"
# file = "result_ssad_p_ratio.out"
#
dict_ = {}

for data in ["mnist"]:
    dict_[data] = {}
    file_names = os.listdir(path+data+'-ours')
    for file_name in file_names:
        normal_class = int(file_name.split("_")[3])
        meta_scale = file_name.split("_")[5]
        r = open(path+data+'-ours/'+file_name).read().split("\n")[-2].split(":")[-1]
        if meta_scale not in dict_[data]:
            dict_[data][meta_scale] = 0
        dict_[data][meta_scale] += float(r[2:-1])

for data in ["mnist"]:
    print (data, dict_[data])

