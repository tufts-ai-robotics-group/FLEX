#!/bin/bash 

aff_cf_path=$1 
obj_name=$2 
# --prop_save_dir outputs/grasp_proposals/camera_frame_proposals/camera_frame_${obj_name}_grasp.npz
echo "Getting camera frame proposale for object $obj_name" 
conda run -n cgn python grasps/aograsp/contact_graspnet/contact_graspnet/run_cgn_on_heatmap_file.py --heatmap_file $aff_cf_path --prop_save_dir outputs/grasp_proposals/camera_frame_proposals