#bin/bash!

###############################################################################
# This file tests the effect of the number of labeled training samples on the
# performance of multiple deep outlier detection methods
# To run set your device_number below and execute `bash ratio-labeled-data.sh`
# If you are using a virutal environment you must activate it before running
###############################################################################

device_number="2" # specify GPU ID for cuda execution, check IDs using `nvidia-smi`
output_path="ratio-labeled-data" # top level directory for results

###############################################################################
# The following tests Deep AutoEncoder approaches to outlier detection
###############################################################################

file_path="../main_AE.py"
csv_out=${output_path}/${output_path}-AE.csv
echo "Testing Deep AutoEncoder Methods"
i=0

# FMNIST and SVHN
#for dataset in "mnist"; do
#    for method in "ours"; do
#        mkdir -p ${output_path}/${dataset}-${method}-AE
#        for ratio_polluted in 0.5; do
#            for normal_class in 0 1 2 3 4 5 6 7 8 9; do
#                for num_labels in 20 40 60 80 100; do # 5 points with intervals of 45
#                    echo "Deep AutoEncoder" Iteration $i / 680
#                    CUDA_VISIBLE_DEVICES=${device_number} python3 $file_path $method $dataset $normal_class $ratio_polluted $num_labels $csv_out > ${output_path}/${dataset}-${method}-AE/${dataset}_${ratio_polluted}_${num_labels}_${normal_class}.out
#                  i=$(( i + 1 ))
#               done
#           done
#       done
#    done
#done

# CIFAR
dataset="cifar"
for method in "ours"; do
    mkdir -p ${output_path}/${dataset}-${method}-AE
    for ratio_polluted in 0.1 0.5; do
        for normal_class in 8 9; do
            for num_labels in 100 200 300 400 500; do # 7 points with intervals of 75
                echo "Deep AutoEncoder" Iteration $i / 680
                CUDA_VISIBLE_DEVICES=${device_number} python3 $file_path $method $dataset $normal_class $ratio_polluted $num_labels $csv_out > ${output_path}/${dataset}-${method}-AE/${dataset}_${ratio_polluted}_${num_labels}_${normal_class}.out
                i=$(( i + 1 ))
            done
        done
    done
done

###############################################################################
# End of script
###############################################################################

exit 0
