import os
from filled_the_holes import apply_postprocessing_to_folder

if __name__=='__main__':
    input_folder = '/data/datasets/zheyuan/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task047_AMOS/nnUNetTrainerV2_balance__nnUNetPlansv2.1/predict2'
    output_folder = '/data/datasets/zheyuan/nnUNet/nnUNet_trained_models/nnUNet/3d_fullres/Task047_AMOS/nnUNetTrainerV2_balance__nnUNetPlansv2.1/predict_filled'
    for_which_classes = list(range(1, 16))
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    apply_postprocessing_to_folder(input_folder=input_folder, output_folder=output_folder, 
                                   for_which_classes=for_which_classes,
                                   min_valid_object_size=None, num_processes=32)