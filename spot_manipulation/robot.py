#!/usr/bin/env python3
"""
Spot Robot Interface and Main Interactive Manipulation System

Complete pipeline: Grasp → Wiggle → Estimate Joint → Load Policy → Execute Manipulation
"""

import argparse
import sys
import time
import numpy as np
import cv2

# Spot SDK imports
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.image import ImageClient
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, get_a_tform_b, HAND_FRAME_NAME

# Our modules
from interactive_perception import InteractivePerception
from policy_manager import PolicyManager

# Global for image click
g_image_click = None
g_image_display = None


def cv_mouse_callback(event, x, y, flags, param):
    """Mouse callback for object selection."""
    global g_image_click, g_image_display
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)


class SpotRobotInterface:
    """Clean interface for Spot robot operations."""
    
    def __init__(self, hostname):
        """Initialize robot connection."""
        self.hostname = hostname
        self.setup_robot()
    
    def setup_robot(self):
        """Initialize robot connection and clients."""
        print("🤖 Connecting to Spot...")
        
        # Create SDK and robot
        sdk = bosdyn.client.create_standard_sdk('SpotManipulation')
        self.robot = sdk.create_robot(self.hostname)
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()
        
        # Verify capabilities
        assert self.robot.has_arm(), "Robot must have an arm!"
        
        # Create clients
        self.lease_client = self.robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
        self.robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.manipulation_api_client = self.robot.ensure_client(ManipulationApiClient.default_service_name)
        self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
        
        print("Robot connected successfully")
    
    def power_on_and_stand(self):
        """Power on robot and stand up."""
        print("⚡ Powering on robot...")
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on()
        
        print("🚶 Standing up...")
        blocking_stand(self.command_client, timeout_sec=10)
    
    def power_off(self):
        """Power off robot safely."""
        print("⏸️  Powering down...")
        self.robot.power_off(cut_immediately=False, timeout_sec=20)
    
    def get_gripper_position(self):
        """Get current gripper position and orientation."""
        robot_state = self.robot_state_client.get_robot_state()
        transforms_snapshot = robot_state.kinematic_state.transforms_snapshot
        
        try:
            hand_transform = get_a_tform_b(transforms_snapshot, VISION_FRAME_NAME, HAND_FRAME_NAME)
            position = np.array([hand_transform.x, hand_transform.y, hand_transform.z])
            return position, hand_transform.rot
        except:
            print("⚠️  Could not get hand transform")
            return None, None
    
    def move_arm_to_position(self, target_position, duration=1.0):
        """Move arm to target position."""
        current_pos, current_rot = self.get_gripper_position()
        if current_pos is None or current_rot is None:
            return False
        
        cmd = RobotCommandBuilder.arm_pose_command(
            target_position[0], target_position[1], target_position[2],
            current_rot.w, current_rot.x, current_rot.y, current_rot.z,
            VISION_FRAME_NAME, duration
        )
        
        end_time = time.time() + duration + 1.0
        self.command_client.robot_command(cmd, end_time_secs=end_time)
        return True
    
    def execute_grasp(self):
        """Execute object grasping with visual selection."""
        global g_image_click, g_image_display
        
        print("📷 Taking image for object selection...")
        
        # Get image
        image_responses = self.image_client.get_image_from_sources(['frontleft_fisheye_image'])
        if len(image_responses) != 1:
            print("❌ Failed to get image")
            return False
            
        image = image_responses[0]
        
        # Convert image
        if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16
        else:
            dtype = np.uint8
        img = np.fromstring(image.shot.image.data, dtype=dtype)
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)
        
        # Show image and wait for click
        print("👆 Click on object to grasp...")
        cv2.namedWindow('Select Object')
        cv2.setMouseCallback('Select Object', cv_mouse_callback)
        
        g_image_click = None
        g_image_display = img
        cv2.imshow('Select Object', img)
        
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
        
        cv2.destroyAllWindows()
        
        print(f"🎯 Selected point: {g_image_click}")
        
        # Create grasp request
        pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole
        )
        
        # Send grasp request
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)
        cmd_response = self.manipulation_api_client.manipulation_api_command(grasp_request)
        
        # Wait for grasp completion
        print("🤏 Executing grasp...")
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id
            )
            response = self.manipulation_api_client.manipulation_api_feedback_command(feedback_request)
            
            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED:
                print(" Grasp successful!")
                return True
            elif response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                print(" Grasp failed!")
                return False
            
            time.sleep(0.5)


class SpotManipulationSystem:
    """Main system orchestrating the complete manipulation pipeline."""
    
    def __init__(self, hostname, models_dir="models"):
        """Initialize the system."""
        self.robot_interface = SpotRobotInterface(hostname)
        self.interactive_perception = InteractivePerception()
        self.policy_manager = PolicyManager(models_dir)
        
        self.initial_position = None
    
    def execute_wiggling_and_analyze(self):
        """Execute wiggling movements and analyze trajectory."""
        print(" Starting interactive perception (wiggling)...")
        
        # Get initial position
        self.initial_position, _ = self.robot_interface.get_gripper_position()
        
        # Generate wiggle positions
        wiggle_positions = self.interactive_perception.generate_wiggle_positions(self.initial_position)
        
        trajectory = []
        
        for i, target_pos in enumerate(wiggle_positions):
            print(f"   Moving to position {i+1}/{len(wiggle_positions)}")
            
            # Move to position
            self.robot_interface.move_arm_to_position(target_pos, duration=2.0)
            
            # Record trajectory during movement
            for _ in range(10):  # Record 10 points per movement
                pos, _ = self.robot_interface.get_gripper_position()
                if pos is not None:
                    trajectory.append(pos.copy())
                time.sleep(0.2)
        
        print(f"✅ Recorded trajectory with {len(trajectory)} points")
        
        # Analyze trajectory and estimate joint
        joint_type, joint_params = self.interactive_perception.analyze_trajectory_and_estimate_joint(
            np.array(trajectory)
        )
        
        # Load appropriate policy
        self.policy_manager.load_policy(joint_type)
        
        return True
    
    def execute_manipulation_policy(self, max_steps=30):
        """Execute manipulation using loaded policy."""
        print(" Executing manipulation policy...")
        
        for step in range(max_steps):
            # Get current state
            current_pos, _ = self.robot_interface.get_gripper_position()
            if current_pos is None:
                continue
            
            # Build state vector
            state = self.interactive_perception.construct_state_vector(current_pos, self.initial_position)
            
            # Execute policy step
            action, target_pos = self.policy_manager.execute_policy_step(
                state, current_pos, self.robot_interface, action_scale=0.02
            )
            
            displacement = current_pos - self.initial_position
            print(f"   Step {step}: action={action}, displacement={displacement}")
            
            time.sleep(1.5)
        
        print(" Manipulation completed!")
    
    def run_complete_pipeline(self):
        """Run the complete interactive manipulation pipeline."""
        try:
            with bosdyn.client.lease.LeaseKeepAlive(
                self.robot_interface.lease_client, must_acquire=True, return_at_exit=True
            ):
                # Power on and stand
                self.robot_interface.power_on_and_stand()
                
                # Execute pipeline
                if self.robot_interface.execute_grasp():
                    if self.execute_wiggling_and_analyze():
                        self.execute_manipulation_policy()
                
                # Power down
                self.robot_interface.power_off()
                
        except Exception as e:
            print(f"❌ Error: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Spot Interactive Manipulation')
    parser.add_argument('hostname', help='Hostname or IP of Spot robot')
    parser.add_argument('--models-dir', default='models', help='Directory with policy models')
    
    args = parser.parse_args()
    
    try:
        # Create and run system
        system = SpotManipulationSystem(args.hostname, args.models_dir)
        system.run_complete_pipeline()
        print("🎉 Mission accomplished!")
        return True
    except Exception as e:
        print(f"💥 System failed: {e}")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)