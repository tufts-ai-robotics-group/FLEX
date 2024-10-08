import ctypes
import subprocess
import os
import sys

# TODO:set work dir for GPD c++ section so that it can find model
# currently only working when run python program in gpd/build folder

sys.path.append("usr/local/lib/")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class HandPose(ctypes.Structure):
    _fields_=[("size", ctypes.c_int), ("pos_x", ctypes.c_float*100),
              ("pos_y", ctypes.c_float*100), ("pos_z", ctypes.c_float*100), 
              ("grasp_vector_list", (ctypes.c_float*3)*100),
              ("rotation_matrix_list", (ctypes.c_float*9)*100),] 
    
    # def __str__(self) -> str:
    #     return 'size: ' + 

def gpd_get_grasp_pose(pcd_file_name, cfg_file_name='eigen_params.cfg'):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
    print(base_dir)
    # base_dir_gpd = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
    # print(base_dir_gpd)
    work_dir = os.path.join(base_dir,"gpd/build/")
    cfg_dir = os.path.join(base_dir,"cfg/") # Config files are under FRL directory
    pcd_dir = os.path.join(base_dir, "point_clouds/")
    cfg_file = os.path.join(cfg_dir, cfg_file_name)
    pcd_file = os.path.join(pcd_dir,pcd_file_name)
    print(cfg_file, pcd_file)
    handle = ctypes.CDLL(work_dir+"gpd_cips_lib.so")
    func = handle.get_pose
    func.argtypes = [ctypes.POINTER(ctypes.c_char_p)]
    func.restype = HandPose
    dirs = ["./cips_grasp".encode("utf-8"),
        cfg_file.encode("utf-8"),
        pcd_file.encode("utf-8")]
    argv = (ctypes.c_char_p * len(dirs))("./cips_grasp".encode("utf-8"), cfg_file.encode("utf-8"), pcd_file.encode("utf-8"))
    # argv[:] = dirs
    print(argv)
    argc = ctypes.c_int(len(argv))
    # grasp_poses_c_type = HandPose()
    print(len(argv))
    handle.main(argc, argv)
    grasp_poses_c_type = handle.get_pose(argv) 
    return grasp_poses_c_type
    # print(grasp_poses_c_type.size)
    # grasp_poses_c_type = myfunction(argc, argv)


