import numpy as np
from sklearn.decomposition import PCA
from scipy.optimize import least_squares


class InteractivePerception:
    """
    Interactive perception module that handles:
    1. Generating wiggle movements
    2. Analyzing trajectory to estimate joint type and parameters
    3. Constructing state vectors for policy input
    """
    
    def __init__(self, movement_distance=0.05):
        """
        Args:
            movement_distance: Distance to move in each direction (meters)
        """
        self.movement_distance = movement_distance
        self.joint_params = None
        self.joint_type = None
    
    def generate_wiggle_positions(self, start_position):
        """
        Generate target positions for 4-directional wiggling exploration.
        
        Args:
            start_position: Initial gripper position [x, y, z]
            
        Returns:
            list: Target positions for wiggling sequence
        """
        positions = [start_position.copy()]  # Start position
        
        # 4 directions: forward, backward, left, right
        directions = [
            np.array([self.movement_distance, 0, 0]),   # Forward
            np.array([-self.movement_distance, 0, 0]),  # Backward  
            np.array([0, self.movement_distance, 0]),   # Left
            np.array([0, -self.movement_distance, 0])   # Right
        ]
        
        for direction in directions:
            # Move to direction
            target = start_position + direction
            positions.append(target.copy())
            
            # Return to center
            positions.append(start_position.copy())
        
        return positions
    
    def prismatic_error_analysis(self, trajectory):
        """Calculate prismatic joint error and axis."""
        centroid = np.mean(trajectory, axis=0)
        X = trajectory - centroid
        _, _, Vt = np.linalg.svd(X)
        line_direction = Vt[0]
        projections = np.dot(X, line_direction[:, np.newaxis]) * line_direction
        residuals = np.linalg.norm(X - projections, axis=1) 
        ss_residuals = np.sum(residuals**2) / len(trajectory)
        return ss_residuals, line_direction
    
    def revolute_error_analysis(self, trajectory):
        """Calculate revolute joint error and parameters."""
        mean = np.mean(trajectory, axis=0) 
        X = trajectory - mean
        pca = PCA(n_components=3)
        pca.fit(X)

        normal_vector = pca.components_[-1]
        
        # Simple circle fitting
        def residuals(params, points):
            center = np.array([params[0], params[1], params[2]])
            radius = params[3]
            distance = np.linalg.norm(points - center, axis=1) - radius
            return distance
        
        # Initial estimate
        initial_center = mean
        initial_radius = np.std(np.linalg.norm(X, axis=1))
        initial_guess = np.concatenate([initial_center, [initial_radius]])
        
        result = least_squares(residuals, initial_guess, args=(trajectory,))
        center = result.x[:-1]
        radius = result.x[-1]
        
        # Calculate MSE
        distances = np.linalg.norm(trajectory - center, axis=1)
        mse = np.mean((distances - radius) ** 2)
        
        return mse, center, radius, normal_vector
    
    def analyze_trajectory_and_estimate_joint(self, trajectory):
        """
        Analyze trajectory to estimate joint type and parameters.
        
        Args:
            trajectory: Array of positions [N x 3]
            
        Returns:
            tuple: (joint_type, joint_params)
        """
        # Calculate errors for both joint types
        prismatic_error, prismatic_axis = self.prismatic_error_analysis(trajectory)
        revolute_error, revolute_center, revolute_radius, revolute_axis = self.revolute_error_analysis(trajectory)
        
        print(f"Prismatic error: {prismatic_error:.6f}")
        print(f"Revolute error: {revolute_error:.6f}")
        
        # Select joint type based on lower error
        if prismatic_error < revolute_error:
            self.joint_type = "prismatic"
            self.joint_params = {"axis": prismatic_axis, "error": prismatic_error}
            print("→ Estimated joint type: PRISMATIC")
        else:
            self.joint_type = "revolute" 
            self.joint_params = {
                "center": revolute_center,
                "radius": revolute_radius, 
                "axis": revolute_axis,
                "error": revolute_error
            }
            print("→ Estimated joint type: REVOLUTE")
        
        return self.joint_type, self.joint_params
    
    def construct_state_vector(self, current_position, initial_position):
        """
        Construct state vector for policy input.
        
        Args:
            current_position: Current gripper position [x, y, z] 
            initial_position: Initial grasp position [x, y, z]
            
        Returns:
            np.array: State vector [joint_axis_x, joint_axis_y, joint_axis_z, 
                                   displacement_x, displacement_y, displacement_z]
        """
        if self.joint_params is None:
            raise ValueError("Must analyze trajectory first!")
        
        # Calculate displacement from initial position
        displacement = current_position - initial_position
        
        # Get normalized joint axis
        joint_axis = self.joint_params["axis"]
        joint_axis = joint_axis / np.linalg.norm(joint_axis)
        
        # Construct state: [joint_axis (3D), displacement (3D)]
        state = np.concatenate([joint_axis, displacement])
        
        return state.astype(np.float32)