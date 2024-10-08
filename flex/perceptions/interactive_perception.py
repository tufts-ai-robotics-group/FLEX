import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 
from sklearn.decomposition import PCA 
from scipy.optimize import minimize
from scipy.optimize import least_squares


def project_points_onto_plane(points, normal_vector, origin):
    projected_points = [] 
    for p in points:
        v = p - origin 
        distance = np.dot(v, normal_vector)
        projected_point = p - distance * normal_vector
        projected_points.append(projected_point)
    return np.array(projected_points)


class InteractivePerception:
    '''
    Estimates the axis and type of the joint given a trajectory of the end effector.
    '''
    def __init__(self, trajectory, logvar=0.1):
        self.trajectory = trajectory 
        self.prismatic_model = LinearRegression()

    def prismatic_error_new(self):
        centroid = np.mean(self.trajectory, axis=0)
        X = self.trajectory - centroid
        _, _, Vt = np.linalg.svd(X)
        line_direction = Vt[0]
        projections = np.dot(X, line_direction[:, np.newaxis]) * line_direction
        residuals = np.linalg.norm(X - projections, axis=1) 
        ss_residuals = np.sum(residuals**2) / len(self.trajectory)
        return ss_residuals, line_direction

    def prismatic_error(self):
        # print(self.trajectory)
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

    # def revolute_error(self):
    #     mean = np.mean(self.trajectory, axis=0) # mean is assumed to be on the plane
    #     centered_points = self.trajectory - mean

    #     pca = PCA(n_components=3) 
    #     pca.fit(centered_points)

    #     normal_vector = pca.components_[2] 
    #     plane_vectors = pca.components_[:2]
    #     # plane_vectors += mean 

    #     print('estimated axis direction: ', normal_vector) 
    #     print('estimated plane: ', plane_vectors) 

    #     d_squared = np.sum(np.dot((self.trajectory-mean), normal_vector)**2)

    #     # projected_points = project_points_onto_plane(self.trajectory, normal_vector, mean_projection) 

    def project_points_onto_plane(self, points, normal, point_on_plane):
        projected_points = []
        for p in points:
            vector = p - point_on_plane
            distance = np.dot(vector, normal)
            projection = p - distance * normal
            projected_points.append(projection)
        return np.array(projected_points)

    # Step 2: Define the circle fitting function in 3D
    def fit_circle_to_arc(self, points_3d):
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
        
        # Perform PCA to project the 3D points onto a 2D plane

        # A rough initial estimate for the circle center could be the centroid of the arc points
        center_estimate = np.mean(points_2d, axis=0)
        result = minimize(loss_function, center_estimate)
        circle_center_2d = result.x
        radius = calc_R(circle_center_2d).mean()

        # Map the 2D center back to 3D using the PCA components
        circle_center_3d = pca.inverse_transform(circle_center_2d)

        return circle_center_3d, radius

    def residuals(self,params, points):
        center = np.array([params[0], params[1], params[2]])
        radius = params[3]
        distance = np.linalg.norm(points - center, axis=1) - radius
        return distance
    
    def compute_mse(self, center, radius):
        distances = np.linalg.norm(self.trajectory - center, axis=1)
        squared_errors = (distances - radius) ** 2
        mse = np.mean(squared_errors)
        return mse
    
    def revolute_error(self): 
        '''
        Returns the posterior error of the revolute joint estimation and the estimated parameters
        error, center, radius, direction
        '''
        mean = np.mean(self.trajectory, axis=0) 
        X = self.trajectory - mean
        pca = PCA(n_components=3)
        pca.fit(X)

        # Normal vector and point on the plane
        normal_vector = pca.components_[-1]
        hinge_axis = pca.components_[1]
        # print('estimated axis direction: ', hinge_axis)
        point_on_plane = np.mean(X, axis=0)

        # Project points onto the plane
        projected_points = project_points_onto_plane(X, normal_vector, point_on_plane)

        # Step 4: Fit the circle to the projected points
        circle_center_3d, radius = self.fit_circle_to_arc(projected_points)
        radius = min(radius, 2)
        circle_center_3d = circle_center_3d + mean

        # print("Estimated Circle Center in 3D:", circle_center_3d + mean)
        # print("Estimated Radius:", radius) 

        p1 = self.trajectory[0]
        p2 = self.trajectory[-1]
        mid_point = (p1 + p2) / 2
        hinge_position = mid_point + np.cross(hinge_axis, p1-p2)
        bounds = ([circle_center_3d[0], circle_center_3d[1], circle_center_3d[2], 0], 
              [circle_center_3d[0] + 10, circle_center_3d[1] + 10, circle_center_3d[2] + 10, 3])
        # print('new estimated hinge position: ', hinge_position)

        result = least_squares(self.residuals, np.concatenate([circle_center_3d, np.array([radius])]), args=(self.trajectory,), bounds=bounds)
        center = result.x[:-1]
        mse = self.compute_mse(result.x[:3], result.x[3])
        # print('result: ', result.x)
        # print('cost: ', result.cost/len(self.trajectory))
        # return result.cost*2/len(self.trajectory), result.x[:-1], result.x[-1], hinge_axis 
        return mse, center, radius, normal_vector
    

if __name__ == '__main__': 
    angles = np.linspace(0, np.pi/18, 80) 
    radius = 1
    trajectory = np.zeros((80, 3)) 
    trajectory = np.random.randn(80, 3) * 0.001
    trajectory[:, 0] += np.cos(angles) * radius 
    trajectory[:, 1] += np.sin(angles) * radius

    trajectory = np.array([[-0.23202173, -0.26965309, 0.8504034],
                      [-0.2321651,  -0.27008291, 0.85040419],
                      [-0.23236463, -0.27068166, 0.85040584],
                      [-0.23263151, -0.27145991, 0.85040576],
                      [-0.23295326, -0.27238933, 0.85040287],
                      [-0.23329967, -0.27340952, 0.85040301],
                      [-0.23365802, -0.27449596, 0.85040783],
                      [-0.23403246, -0.27565752, 0.85041526],
                      [-0.23442858, -0.27690506, 0.85042491],
                      [-0.23484263, -0.27822952, 0.85043643],
                      [-0.23527028, -0.27962006, 0.85044914],
                      [-0.23570764, -0.28106673, 0.85046216],
                      [-0.23615107, -0.28255938, 0.85047529],
                      [-0.2365975,  -0.28408924, 0.85048831],
                      [-0.23704436, -0.28564916, 0.85050067],
                      [-0.2374892,  -0.28723193, 0.85051212],
                      [-0.2379298,  -0.28883077, 0.85052251],
                      [-0.23836424, -0.29043948, 0.85053175],
                      [-0.23879073, -0.29205221, 0.85053977],
                      [-0.23920763, -0.29366367, 0.85054647]])
    ip = InteractivePerception(trajectory)
    error = ip.prismatic_error() 
    # print(trajectory)
    print('prismatic error: ', error) 
    pris_error, c, r, h = ip.revolute_error() 
    print('revolute error: ', pris_error)
    print('estimated center: ', c) 
    print('estimated radius: ', r) 
    print('estimated axis: ', h)