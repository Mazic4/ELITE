#bin/bash!

###############################################################################
# This file tests the effect of the ratio of anomalies in the labeled training
# set on theperformance of multiple deep outlier detection methods
# To run set your device_number below and execute `bash ratio-labeled-data.sh`
# If you are using a virutal environment you must activate it before running
###############################################################################

output_path="fully-supervised"
device_number="0" # specify GPU ID for cuda execution, check IDs using `nvidia-smi`

###############################################################################
# The following tests Deep SVM approaches to outlier detection
###############################################################################

file_path="../fully_supervised.py"
echo "Testing Deep SVM Methods"
i=0

for dataset in "cifar" "mnist" "fmnist"; do
    for method in "ours"; do
        mkdir -p ${output_path}/${dataset}-${method}
        for ratio_polluted in 0.1 0.2 0.3 0.4 0.5; do # 0.05 - 0.5 in 10 intervals of 0.05
            for normal_class in 0 1 2 3 4 5 6 7 8 9; do
                num_labels=2
                if [ "$dataset" = "fmnist" ]; then
                    num_labels=2
                fi
                echo "Deep SVM" Iteration $i / 600
                CUDA_VISIBLE_DEVICES=${device_number} python3 $file_path $method $normal_class $dataset $ratio_polluted $num_labels > ${output_path}/${dataset}-${method}/${dataset}_${ratio_polluted}_${num_labels}_${normal_class}.out
                i=$(( i + 1 ))
            done
        done
    done
done

###############################################################################
# End of script
###############################################################################

exit 0
