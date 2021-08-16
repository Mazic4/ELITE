import numpy as np
import os


path = "ratio-labeled-data/"

# # file = "result_isoforest.out"
# file = "result_ssad_p_ratio.out"
#
dict_ = {}

for data in ["cifar"]:
    dict_[data] = {}
    file_names = os.listdir(path+data+'-ours-AE')
    for file_name in file_names:
        normal_class = int(file_name.split("_")[3][0])
        num_labels = int(file_name.split("_")[2])
        p_ratio = file_name.split("_")[1]
        try:
            r = open(path+data+'-ours-AE/'+file_name).read().split("\n")[-2].split(":")[-1]
        except:
            print (normal_class, p_ratio, num_labels)
            continue
        if p_ratio not in dict_[data]:
            dict_[data][p_ratio] = {}
        # if float(r[2:-1]) < 0.8:
        #     print (data, p_ratio, normal_class, r)
        dict_[data][p_ratio][num_labels] = dict_[data][p_ratio].get(num_labels, 0) + float(r)/10

for data in ["cifar"]:
    print (data, dict_[data])

