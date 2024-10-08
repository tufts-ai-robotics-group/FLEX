#include <stdio.h>
#include <string>

#include <gpd/grasp_detector.h>
typedef struct HandPose{
        int size;
        float pos_x[100];
        float pos_y[100];
        float pos_z[100];
        float grasp_vector_list[100][3];
        float rotation_matrix_list[100][9];
    }HandPose;

namespace gpd{
namespace grasp_cips_space{

    

    bool checkFileExists(const std::string &file_name) {
    std::ifstream file;
    file.open(file_name.c_str());
    if (!file) {
        std::cout << "File " + file_name + " could not be found!\n";
        return false;
    }
    file.close();
    return true;
    }

    HandPose DoMain(int argc, char *argv[]) {
    // Read arguments from command line.
    if (argc < 3) {
        std::cout << "Error: Not enough input arguments!\n\n";
        std::cout << "Usage: detect_grasps CONFIG_FILE PCD_FILE [NORMALS_FILE]\n\n";
        std::cout << "Detect grasp poses for a point cloud, PCD_FILE (*.pcd), "
                    "using parameters from CONFIG_FILE (*.cfg).\n\n";
        std::cout << "[NORMALS_FILE] (optional) contains a surface normal for each "
                    "point in the cloud (*.csv).\n";
        exit(1);

    }

    std::string config_filename = argv[1];
    std::string pcd_filename = argv[2];
    if (!checkFileExists(config_filename)) {
        printf("Error: config file not found!\n");
        exit(1);
    }
    if (!checkFileExists(pcd_filename)) {
        printf("Error: PCD file not found!\n");
        exit(1);
    }

    // Read parameters from configuration file.
    util::ConfigFile config_file(config_filename);
    config_file.ExtractKeys();

    // Set the camera position. Assumes a single camera view.
    std::vector<double> camera_position =
        config_file.getValueOfKeyAsStdVectorDouble("camera_position",
                                                    "0.0 0.0 0.0");
    Eigen::Matrix3Xd view_points(3, 1);
    view_points << camera_position[0], camera_position[1], camera_position[2];

    // Load point cloud from file.
    util::Cloud cloud(pcd_filename, view_points);
    if (cloud.getCloudOriginal()->size() == 0) {
        std::cout << "Error: Input point cloud is empty or does not exist!\n";
        exit(1);
    }

    // Load surface normals from file.
    if (argc > 3) {
        std::string normals_filename = argv[3];
        cloud.setNormalsFromFile(normals_filename);
        std::cout << "Loaded surface normals from file: " << normals_filename
                << "\n";
    }

    GraspDetector detector(config_filename);

    // Preprocess the point cloud.
    detector.preprocessPointCloud(cloud);

    // If the object is centered at the origin, reverse all surface normals.
    bool centered_at_origin =
        config_file.getValueOfKey<bool>("centered_at_origin", false);
    if (centered_at_origin) {
        printf("Reversing normal directions ...\n");
        cloud.setNormals(cloud.getNormals() * (-1.0));
    }
    std::vector<std::unique_ptr<candidate::Hand>> clusters;
    // Detect grasp poses.
    HandPose res;
    clusters = detector.detectGrasps(cloud);
    // printf("BOOM");
    float result[clusters.size()*3];
    for (int i = 0; i < clusters.size(); i++){
        std::cout << "Grasp " << i << ": " << clusters[i]->getScore() << "\n";
        for(int j = 0; j < 3; j++){
        result[3*i+j] = clusters[i]->getPosition().data()[j];  
        }
        res.pos_x[i] = clusters[i]->getPosition().data()[0];
        res.pos_y[i] = clusters[i]->getPosition().data()[1];
        res.pos_z[i] = clusters[i]->getPosition().data()[2];
        for (int j = 0; j < 3; j++){
                res.grasp_vector_list[i][j] = clusters[i]->getApproach().data()[j];
        }

        // for (int j = 0; j < 3; j++)
        // {
        //     for (int k = 0; k < 9; k++){
        //         res.rotation_matrix_list[i][j][k] = float(clusters[i]->getOrientation().data()[j]);
        //     }
        // }
        for (int j = 0; j < 9; j++)
        {
            res.rotation_matrix_list[i][j] = clusters[i]->getOrientation().data()[j];
        }
        // Eigen::Map<Eigen::Matrix3d>(&res.rotation_matrix_list[i][0][0], 3, 3) = clusters[i]->getOrientation();
        std::cout << "hand positions" << clusters[i]->getPosition() << endl;
        std::cout << "array" << result[3*i]  <<  result[3*i+1]  <<result[3*i+2] << endl;
        std::cout << "rotation matrix" << res.rotation_matrix_list[i] << endl;
    }
    res.size = clusters.size();
    std::cout << "check1" << endl;
    // std::cout << result << endl;
    return res;
}
} 
}

extern "C" HandPose get_pose(char *argv[]) {
    int num = 3;
    std::cout << "entering function" << num << endl;
    std::cout << argv[1]  << endl;
    HandPose result =  gpd::grasp_cips_space::DoMain(num, argv);
//   std::cout << "test!" << endl;
//   float *res = gpd::grasp_cips::get_hand_pos();
//   std::cout << res << endl;
//   std::cout << "free space!" <<endl;
    std::cout << result.pos_x[0] << endl;
    return result;
}

int main(int argc, char *argv[]){
    std::cout << argv[1]  << endl;

    // HandPose result =  gpd::grasp_cips_space::DoMain(argc, argv);
    // doMain();
    return 0;
    // return gpd::grasp_cips_space::DoMain(argc, argv);
}