#bin/bash!

###############################################################################
# This file tests the effect of the ratio of anomalies in the labeled training
# set on theperformance of multiple deep outlier detection methods
# To run set your device_number below and execute `bash ratio-labeled-data.sh`
# If you are using a virutal environment you must activate it before running
###############################################################################

output_path="num-anomaly-classes"
device_number="0" # specify GPU ID for cuda execution, check IDs using `nvidia-smi`

###############################################################################
# The following tests Deep SVM approaches to outlier detection
###############################################################################

file_path="../main.py"
echo "Testing Deep SVM Methods"
i=0

for dataset in "cifar" "svhn" "fmnist"; do
    for method in "baseline" "ours"; do
        mkdir -p ${output_path}/${dataset}-${method}
        for ratio_polluted in 0.1 0.5; do
            for num_outlier_class in 5 6 7 8 9; do
                for normal_class in 0 1 2 3 4 5 6 7 8 9; do
                    num_labels=100
                    if [ "$dataset" = "fmnist" ]; then
                        num_labels=20
                    fi
                    echo "Deep SVM" Iteration $i / 600
                    CUDA_VISIBLE_DEVICES=${device_number} python3 $file_path $method $normal_class $dataset $ratio_polluted $num_labels $num_outlier_class > ${output_path}/${dataset}-${method}/${dataset}_${ratio_polluted}_${num_labels}_${normal_class}_${num_outlier_class}.out
                    i=$(( i + 1 ))
                done
            done
        done
    done
done

###############################################################################
# The following tests Deep AutoEncoder approaches to outlier detection
###############################################################################

file_path="../main_AE.py"
csv_out=${output_path}/${output_path}-AE.csv
echo "Testing Deep AutoEncoder Methods"
i=0

for dataset in "cifar" "svhn" "fmnist"; do
    for method in "baseline" "ours"; do
        mkdir -p ${output_path}/${dataset}-${method}-AE
        for ratio_polluted in 0.1 0.5; do
            for num_outlier_class in 5 6 7 8 9; do
                for normal_class in 0 1 2 3 4 5 6 7 8 9; do
                    num_labels=100
                    if [ "$dataset" = "fmnist" ]; then
                        num_labels=20
                    fi
                    echo "Deep AutoEncoder" Iteration $i / 600
                    CUDA_VISIBLE_DEVICES=${device_number} python3 $file_path $method $dataset $normal_class $ratio_polluted $num_labels $csv_out $num_outlier_class > ${output_path}/${dataset}-${method}-AE/${dataset}_${ratio_polluted}_${num_labels}_${normal_class}_${num_outlier_class}.out
                    i=$(( i + 1 ))
                done
            done
        done
    done
done

###############################################################################
# End of script
###############################################################################

exit 0
