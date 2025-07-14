
# FLEX-real-robot - UR5
This contains instructions on how to run the FLEX code on the UR5 arm to open a drawer.


## How to Power Up the Robot

1. Push the power button on the `teaching pendant` (the tablet).
2. From the warning sign, select `Run to initialization option`.
3. Push `ON` to power up the robotic arm.
4. Press `START` to release the brakes.
5. Push `OK` in the bottom right.

## How to Bring Up the Drivers on the Computer
On the computer connected to the UR5 arm, follow these steps:

1. Open a terminal and run the following commands:
    ```bash
    cd ~/catkin_ws
    source devel/setup.bash
    export ROS_IP=10.3.4.100
    ```

2. Launch the bringup launch file:
    ```bash
    sudo chmod 777 /dev/ttyUSB0
    roslaunch turbo_bringup right_arm.launch
    ```
    You will see the gripper opening and closing, indicating a successful launch!

## Teaching Pendant Setup

Load the driver so that the program can actually move the robot.

1. On the main screen, click `Run Program`.
2. Go to `File --> Load Program`.
3. Select `ur_driver.urp` and click `Open`.
4. Press `Play` (`▶` symbol).
5. Press `continue` on the popup.

Voila! We are done!

## How to Launch the Camera Drivers

On the same computer connected to the UR5, follow these steps:

1. Open a terminal and run the following commands:

    ```bash
    cd ~/catkin_ws
    source devel/setup.bash
    export ROS_IP=10.3.4.100
    ```

    ```bash
    cd ~/catkin_ws/src/ForceRL_driver_bk/launch
    roslaunch realsense.launch
    ```

Now you have successfully launched all the drivers for the UR5 arm and sensors.

## Running FLEX

To run FLEX, follow these steps on the computer connected to the UR5 arm:

This is the drive that receives coordinate for grasping and actuate the grasping.

1. Open a terminal and run:

    
    ```bash
    cd ~/catkin_ws
    source devel/setup.bash
    export ROS_IP=10.3.4.100
    ```

    ```bash
    rosrun forcerl_driver grasp_object_driver.py
    ```
    If you see something like "Ready to take commands for planning group...", you're good to go!

Now switch to the computer that can run the models for FLEX.

### On the Computer Running FLEX Models
We use a different computer (let's call it the second computer) to run FLEX model because the computer connected to UR5 can't.

1. In a terminal, run: 
    ```bash
    cd projects/frl-real/ForceRL-real-robot
    conda deactivate  # because ROS can't work well with conda
    source devel/setup.bash
    ```

2. Set up `ROS_IP` and `ROS_MASTER_URI` to use the `roscore` of the client machine:
    ```bash
    export ROS_IP=10.3.13.148
    export ROS_MASTER_URI=http://10.3.4.100:11311
    ```

> **Note:** Do this step for every terminal you open!

3. Run the point transform to convert camera frame proposals to world frame using the `tf2` package of ROS:
    ```bash
    rosrun frl_rollout point_transform.py
    ```

4. On another terminal, **repeat step 1 and 2**, then runaction:
    ```bash
    rosrun frl_rollout send_action.py
    ```
    This will send the commands to be executed on the arm.

    The terminal with `send_action.py` runnning will show
    ```Press Enter to send the action X```. Press `Enter` the send the target coordinate to the Master computer.

    On the Master computer, once the `grasp_object_driver.py` receive the corrdinate, it will print `Go...`, press `Enter` to allow the actual movement.

    Keep pressing `Enter` on both computers to successfully execute the actions.

## Running the Policy for Opening the Drawer

1. Open a terminal **on the Master computer**, and run:

    
    ```bash
    cd ~/catkin_ws
    source devel/setup.bash
    export ROS_IP=10.3.4.100
    ```

    ```bash
    rosrun forcerl_driver policy_rollout_driver.py
    ```
    If you see something like "Ready to take commands for planning group...", you're good to go!


1. In a new terminal on **the second computer**, activate the conda environment:
    ```bash
    cd projects/frl-real/ForceRL-real-robot
    conda activate frl
    ```

2. Run the policy:
    ```bash
    python src/frl_rollout/scripts/policy_server.py
    ```

3. On another terminal, deactivate conda:
    ```bash
    cd projects/frl-real/ForceRL-real-robot
    conda deactivate
    ```

4. Source the environment:
    ```bash
    source devel/setup.bash
    ```

5. Don't forget to export `ROS_IP` and `ROS_MASTER_URI`:
    ```bash
    export ROS_IP=10.3.13.148
    export ROS_MASTER_URI=http://10.3.4.100:11311
    ```

6. Run the rollout policy:
    ```bash
    rosrun frl_rollout rollout_policy.py
    ```

This will send the commands to be executed on the arm. Keep pressing `Enter` on the Master computers to successfully execute the actions. Kill the running process on the master computer when you are done with opening the drawer. 


## shut down the robot using the learning pendant
1. Press 'STOP' (square symbol) to stop the driver. 
2. press 'file' -> 'exit'
3. Press "shut down robot"

That's all the demo! Thank you!


# Code structure

## pointcloud_seg
This package is for getting the segmented depth image to the target object given a segmentation mask. The segmented depth image is useful for generating segmeted point cloud for grasp proposals.



## depth_image.py
This is a simple script for converting masked depth image generated by ```pointcloud_seg``` into pointcloud.

## frl_rollout
This is the "server" side for the real robot code. It runs the policy and send action command to the client machine which is connected with the robot.

# Running the code
The grasp proposals for drawer are stored in ```real_small_drawer.npz```
## When already having the grasp proposals

1. The usual stuff
```
souce devel/setup.bash
```

2. setup ROS_IP and ROS_MASTER_URI to use the roscore of the cilent machine
```
export ROS_IP=10.3.13.148
export ROS_MASTER_URI=http://10.3.4.100:11311
```

3. Launch the cilent and robot driver codes on the cilent machine

4. run the point_transform to transform the camera frame proposals to world frame using ```tf2``` package of ROS. 
```
rosrun frl_rollout point_transform.py
```
