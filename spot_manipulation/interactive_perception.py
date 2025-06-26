"""
Interactive perception module for joint type and parameter estimation.
Adapted from the original FLEX interactive perception code.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.decomposition import PCA 
from scipy.optimize import minimize
from scipy.optimize import least_squares
from typing import Tuple, Dict


def project_points_onto_plane(points: np.ndarray, normal_vector: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """
    Project 3D points onto a plane defined by normal vector and origin.
    
    Args:
        points: Array of 3D points (N, 3)
        normal_vector: Normal vector of the plane
        origin: Point on the plane
        
    Returns:
        Array of projected points (N, 3)
    """
    projected_points = [] 
    for p in points:
        v = p - origin 
        distance = np.dot(v, normal_vector)
        projected_point = p - distance * normal_vector
        projected_points.append(projected_point)
    return np.array(projected_points)


class InteractivePerception:
    """
    Estimates the axis and type of the joint given a trajectory of the end effector.
    Adapted from the original FLEX implementation.
    """
    
    def __init__(self, trajectory: np.ndarray, logvar: float = 0.1):
        """
        Initialize interactive perception with trajectory data.
        
        Args:
            trajectory: Array of end-effector positions (N, 3)
            logvar: Log variance parameter for noise modeling
        """
        self.trajectory = trajectory 
        self.prismatic_model = LinearRegression()

    def prismatic_error_new(self) -> Tuple[float, np.ndarray]:
        """
        Alternative prismatic error calculation using SVD.
        
        Returns:
            Tuple of (error, line_direction)
        """
        centroid = np.mean(self.trajectory, axis=0)
        X = self.trajectory - centroid
        _, _, Vt = np.linalg.svd(X)
        line_direction = Vt[0]
        projections = np.dot(X, line_direction[:, np.newaxis]) * line_direction
        residuals = np.linalg.norm(X - projections, axis=1) 
        ss_residuals = np.sum(residuals**2) / len(self.trajectory)
        return ss_residuals, line_direction

    def prismatic_error(self) -> Tuple[float, np.ndarray]:
        """
        Calculate prismatic joint error and direction using linear regression.
        
        Returns:
            Tuple of (mean_squared_error, direction_vector)
        """
        X = self.trajectory[:, 1:] 
        y = self.trajectory[:, 0]   
        self.prismatic_model.fit(X, y)

        # Predict and calculate error
        y_pred = self.prismatic_model.predict(X)
        mse = mean_squared_error(y, y_pred)

        a, b = self.prismatic_model.coef_
        c = self.prismatic_model.intercept_ 

        direction = np.array([1, -a, -b])
        direction /= np.linalg.norm(direction)
        return mse, direction

    def project_points_onto_plane(self, points: np.ndarray, normal: np.ndarray, point_on_plane: np.ndarray) -> np.ndarray:
        """
        Project points onto a plane.
        
        Args:
            points: Array of 3D points
            normal: Normal vector of the plane
            point_on_plane: Point on the plane
            
        Returns:
            Array of projected points
        """
        projected_points = []
        for p in points:
            vector = p - point_on_plane
            distance = np.dot(vector, normal)
            projection = p - distance * normal
            projected_points.append(projection)
        return np.array(projected_points)

    def fit_circle_to_arc(self, points_3d: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Fit a circle to 3D points projected onto a 2D plane.
        
        Args:
            points_3d: Array of 3D points
            
        Returns:
            Tuple of (circle_center_3d, radius)
        """
        # Function to fit a circle to 3D points projected onto a 2D plane
        pca = PCA(n_components=2)
        points_2d = pca.fit_transform(points_3d)
        
        def calc_R(c):
            """ Calculate the distance of each 2D point from the center c=(xc, yc) """
            return np.sqrt((points_2d[:, 0] - c[0]) ** 2 + (points_2d[:, 1] - c[1]) ** 2)

        def loss_function(c):
            """ Calculate the loss as the sum of squared differences from the mean radius """
            Ri = calc_R(c)
            return np.sum((Ri - Ri.mean()) ** 2)
        
        # A rough initial estimate for the circle center could be the centroid of the arc points
        center_estimate = np.mean(points_2d, axis=0)
        result = minimize(loss_function, center_estimate)
        circle_center_2d = result.x
        radius = calc_R(circle_center_2d).mean()

        # Map the 2D center back to 3D using the PCA components
        circle_center_3d = pca.inverse_transform(circle_center_2d)

        return circle_center_3d, radius

    def residuals(self, params: np.ndarray, points: np.ndarray) -> np.ndarray:
        """
        Calculate residuals for circle fitting.
        
        Args:
            params: Parameters [center_x, center_y, center_z, radius]
            points: Array of 3D points
            
        Returns:
            Array of residuals
        """
        center = np.array([params[0], params[1], params[2]])
        radius = params[3]
        distance = np.linalg.norm(points - center, axis=1) - radius
        return distance
    
    def compute_mse(self, center: np.ndarray, radius: float) -> float:
        """
        Compute mean squared error for circle fitting.
        
        Args:
            center: Circle center
            radius: Circle radius
            
        Returns:
            Mean squared error
        """
        distances = np.linalg.norm(self.trajectory - center, axis=1)
        squared_errors = (distances - radius) ** 2
        mse = np.mean(squared_errors)
        return mse
    
    def revolute_error(self) -> Tuple[float, np.ndarray, float, np.ndarray]:
        """
        Calculate revolute joint error and parameters.
        
        Returns:
            Tuple of (error, center, radius, normal_vector)
        """
        mean = np.mean(self.trajectory, axis=0) 
        X = self.trajectory - mean
        pca = PCA(n_components=3)
        pca.fit(X)

        # Normal vector and point on the plane
        normal_vector = pca.components_[-1]
        hinge_axis = pca.components_[1]
        point_on_plane = np.mean(X, axis=0)

        # Project points onto the plane
        projected_points = project_points_onto_plane(X, normal_vector, point_on_plane)

        # Fit the circle to the projected points
        circle_center_3d, radius = self.fit_circle_to_arc(projected_points)
        radius = min(radius, 2)  # Cap radius at 2 meters
        circle_center_3d = circle_center_3d + mean

        # Refine circle parameters using least squares
        p1 = self.trajectory[0]
        p2 = self.trajectory[-1]
        mid_point = (p1 + p2) / 2
        hinge_position = mid_point + np.cross(hinge_axis, p1-p2)
        
        bounds = ([circle_center_3d[0], circle_center_3d[1], circle_center_3d[2], 0], 
                  [circle_center_3d[0] + 10, circle_center_3d[1] + 10, circle_center_3d[2] + 10, 3])

        result = least_squares(self.residuals, np.concatenate([circle_center_3d, np.array([radius])]), 
                              args=(self.trajectory,), bounds=bounds)
        center = result.x[:-1]
        mse = self.compute_mse(result.x[:3], result.x[3])
        
        return mse, center, radius, normal_vector

    def estimate_joint_parameters(self) -> Dict:
        """
        Estimate joint type and parameters from trajectory.
        
        Returns:
            Dictionary containing joint parameters
        """
        # Calculate errors for both joint types
        prismatic_error, prismatic_direction = self.prismatic_error()
        revolute_error, revolute_center, revolute_radius, revolute_axis = self.revolute_error()
        
        # Determine joint type based on error comparison
        if prismatic_error < revolute_error:
            joint_type = 'prismatic'
            joint_direction = prismatic_direction
            joint_position = None
            joint_radius = None
        else:
            joint_type = 'revolute'
            joint_direction = revolute_axis
            joint_position = revolute_center
            joint_radius = revolute_radius
        
        return {
            'joint_type': joint_type,
            'joint_direction': joint_direction,
            'joint_position': joint_position,
            'joint_radius': joint_radius,
            'prismatic_error': prismatic_error,
            'revolute_error': revolute_error
        }


if __name__ == '__main__':
    # Test with sample trajectory
    angles = np.linspace(0, np.pi/18, 80) 
    radius = 1
    trajectory = np.zeros((80, 3)) 
    trajectory = np.random.randn(80, 3) * 0.001
    trajectory[:, 0] += np.cos(angles) * radius 
    trajectory[:, 1] += np.sin(angles) * radius

    ip = InteractivePerception(trajectory)
    params = ip.estimate_joint_parameters()
    
    print('Estimated joint parameters:')
    print(f"Joint type: {params['joint_type']}")
    print(f"Joint direction: {params['joint_direction']}")
    print(f"Prismatic error: {params['prismatic_error']}")
    print(f"Revolute error: {params['revolute_error']}") 