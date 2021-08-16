#bin/bash!

###############################################################################
# This file tests the effect of the number of labeled training samples on the
# performance of multiple deep outlier detection methods
# To run set your device_number below and execute `bash ratio-labeled-data.sh`
# If you are using a virutal environment you must activate it before running
###############################################################################

device_number="1" # specify GPU ID for cuda execution, check IDs using `nvidia-smi`
output_path="sensitivity_margin" # top level directory for results

###############################################################################
# The following tests Deep SVM approaches to outlier detection
###############################################################################

file_path="../main.py"
echo "Testing Deep SVM Methods"
i=0

# FMNIST and SVHN
for dataset in "cifar"; do
    for method in "ours"; do
        mkdir -p ${output_path}/${dataset}-${method}
        for ratio_polluted in 0.1; do
            for normal_class in 0 1 2 3 4 5 6 7 8 9; do
                for num_labels in 100 ; do # 5 points with intervals of 45
                    for meta_scale in 1; do
			for margin in 1 5 10 50 100; do
				echo "Deep SVM" Iteration $i
                    		CUDA_VISIBLE_DEVICES=${device_number} python3 $file_path $method $normal_class $dataset $ratio_polluted $num_labels $meta_scale $margin> ${output_path}/${dataset}-${method}/${dataset}_${ratio_polluted}_${num_labels}_${normal_class}_${meta_scale}_${margin}.out
                    		i=$(( i + 1 ))
			done
		    done
                done
            done
        done
    done
done

# CIFAR
#dataset="cifar"
#for method in "baseline" "ours"; do
#    mkdir -p ${output_path}/${dataset}-${method}
#    for ratio_polluted in 0.1 0.5; do
#        for normal_class in 0 1 2 3 4 5 6 7 8 9; do
#            for num_labels in 100 200 300 400 500; do # 7 points with intervals of 75
#                echo "Deep SVM" Iteration $i / 680
#                CUDA_VISIBLE_DEVICES=${device_number} python3 $file_path $method $normal_class $dataset $ratio_polluted $num_labels > ${output_path}/${dataset}-${method}/${dataset}_${ratio_polluted}_${num_labels}_${normal_class}.out
#                i=$(( i + 1 ))
#            done
#        done
#    done
#done

###############################################################################
# End of script
###############################################################################

exit 0
