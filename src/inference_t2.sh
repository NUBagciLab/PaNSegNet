# Add your target domain in the domain_list
# Add the base directory where the target domain is stored
domain_list=(XX)
base_dir="XXX"

for domain in ${domain_list[@]}
do
    echo "Inference on ${domain}"
    input_dir="${base_dir}/${domain}/imagesTs"
    mkdir ${base_dir}/${domain}/inference_segmentation
    OUTPUT_DIRECTORY="${base_dir}/${domain}/inference_segmentation"
    CUDA_VISIBLE_DEVICES=1 nnUNet_predict -tr nnTransUNetTrainerV2 \
     -i ${input_dir} -o ${OUTPUT_DIRECTORY} -t 111 -m 3d_fullres --folds 0
    # CUDA_VISIBLE_DEVICES=1 nnUNet_evaluate_folder -ref ${base_dir}/${domain}/labelsTr -pred ${OUTPUT_DIRECTORY} -l 1
done
