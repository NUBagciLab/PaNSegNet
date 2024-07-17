#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import ast
from copy import deepcopy
from multiprocessing.pool import Pool

import numpy as np
from nnunet.configuration import default_num_threads
from nnunet.evaluation.evaluator import aggregate_scores
from scipy.ndimage import label, binary_fill_holes
import SimpleITK as sitk
from nnunet.utilities.sitk_stuff import copy_geometry
from batchgenerators.utilities.file_and_folder_operations import *
import shutil


def load_remove_save(input_file: str, output_file: str, for_which_classes: list,
                     minimum_valid_object_size: dict = None):
    # Only objects larger than minimum_valid_object_size will be removed. Keys in minimum_valid_object_size must
    # match entries in for_which_classes
    img_in = sitk.ReadImage(input_file)
    img_npy = sitk.GetArrayFromImage(img_in)
    volume_per_voxel = float(np.prod(img_in.GetSpacing(), dtype=np.float64))

    image, largest_removed, kept_size = filled_connected_component(img_npy, for_which_classes,
                                                                   volume_per_voxel,
                                                                   minimum_valid_object_size)
    # print(input_file, "kept:", kept_size)
    img_out_itk = sitk.GetImageFromArray(image)
    img_out_itk = copy_geometry(img_out_itk, img_in)
    sitk.WriteImage(img_out_itk, output_file)
    return largest_removed, kept_size


def filled_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    is_filled_hole = True
    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = np.sum(image==c)

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if is_filled_hole:
                    index_mask = (lmap == object_id) & mask
                    index_mask = binary_fill_holes(index_mask)
                    image[index_mask] = c
            kept_size[c] = np.sum(image==c) - kept_size[c]
    return image, largest_removed, kept_size


def apply_postprocessing_to_folder(input_folder: str, output_folder: str, for_which_classes: list,
                                   min_valid_object_size:dict=None, num_processes=8):
    """
    applies removing of all but the largest connected component to all niftis in a folder
    :param min_valid_object_size:
    :param min_valid_object_size:
    :param input_folder:
    :param output_folder:
    :param for_which_classes:
    :param num_processes:
    :return:
    """
    maybe_mkdir_p(output_folder)
    p = Pool(num_processes)
    nii_files = subfiles(input_folder, suffix=".nii.gz", join=False)
    input_files = [join(input_folder, i) for i in nii_files]
    out_files = [join(output_folder, i) for i in nii_files]
    test = load_remove_save(input_files[1], out_files[1], for_which_classes, min_valid_object_size)
    print(test)
    results = p.starmap_async(load_remove_save, zip(input_files, out_files, [for_which_classes] * len(input_files),
                                                    [min_valid_object_size] * len(input_files)))
    res = results.get()
    p.close()
    p.join()


if __name__ == "__main__":
    input_folder = "/media/fabian/DKFZ/predictions_Fabian/Liver_and_LiverTumor"
    output_folder = "/media/fabian/DKFZ/predictions_Fabian/Liver_and_LiverTumor_postprocessed"
    for_which_classes = [(1, 2), ]
    apply_postprocessing_to_folder(input_folder, output_folder, for_which_classes)
