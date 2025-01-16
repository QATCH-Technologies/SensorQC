import time
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from robot import Robot
from dino_lite_edge import Camera, Microscope
from scipy.spatial import Delaunay
from constants import SystemConstants, RobotConstants, CameraConstants, MicroscopeConstants


class TileScanner:
    def __init__(self, robot_port="COM4", initial_position=SystemConstants.INITIAL_POSITION,
                 final_position=SystemConstants.FINAL_POSITION, x_delta=SystemConstants.X_DELTA, y_delta=SystemConstants.Y_DELTA):
        self.initial_position = initial_position
        self.final_position = final_position
        self.x_min, self.y_max, self.z_fixed, _ = initial_position
        self.x_max, self.y_min, _, _ = final_position
        self.x_delta = x_delta
        self.y_delta = y_delta
        self.scope = Microscope()
        self.cam = Camera()
        self.rob = Robot(port=robot_port, debug=False)

    def ceildiv(self, a: float, b: float) -> int:
        """Performs ceiling division."""
        return int(-(a // -b))

    def interpolate_focus_plane(self, z_points):
        """
        Interpolates a plane based on any number of z-height points.

        Parameters:
            z_points (list of tuples): List of (x, y, z) points defining the plane.

        Returns:
            np.ndarray: Interpolated plane as a 2D array.
        """
        # Extract x, y, z coordinates from the input points
        z_points = np.array(z_points)
        x_coords, y_coords, z_coords = z_points[:,
                                                0], z_points[:, 1], z_points[:, 2]

        # Generate the grid for the interpolation
        x_range = np.linspace(self.x_min, self.x_max, self.ceildiv(
            abs(self.x_max - self.x_min), self.x_delta))
        y_range = np.linspace(self.y_min, self.y_max, self.ceildiv(
            abs(self.y_max - self.y_min), abs(self.y_delta)))
        grid_x, grid_y = np.meshgrid(x_range, y_range)

        # Perform Delaunay triangulation
        tri = Delaunay(z_points[:, :2])

        # Interpolate the z-values for the grid
        plane = np.zeros_like(grid_x)
        simplex = tri.find_simplex(np.c_[grid_x.ravel(), grid_y.ravel()])
        vertices = tri.simplices[simplex]
        bary = tri.transform[simplex, :2].dot(np.c_[grid_x.ravel(
        ) - tri.transform[simplex, 2, 0], grid_y.ravel() - tri.transform[simplex, 2, 1]].T).T
        bary = np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))

        valid = (simplex >= 0) & (bary.min(axis=1) >= 0)
        interp_z = np.zeros(grid_x.size)
        interp_z[valid] = np.sum(
            bary[valid] * z_coords[vertices[valid]], axis=1)
        plane.ravel()[valid] = interp_z[valid]
        return plane

    def init_params(self):
        """Initializes camera and robot to start position."""
        self.rob.begin()
        self.rob.home()
        self.rob.absolute_mode()
        self.scope.disable_microtouch()
        self.scope.led_on(state=MicroscopeConstants.BRIGHT_FIELD)
        x, y, z, _ = self.initial_position
        self.rob.go_to(x, y, z)
        self.scope.set_autoexposure(CameraConstants.AUTOEXPOSURE_OFF)
        self.scope.set_exposure(CameraConstants.AUTOEXPOSURE_VALUE)
        print("Camera has reached the initial position.")
        input("Press Enter to continue...")

    def process_video(self, folder, z_plane):
        """Captures images at each tile location."""
        tile = 1
        num_rows, num_cols = z_plane.shape
        x_values = np.linspace(self.x_min, self.x_max, num_rows)
        y_values = np.linspace(self.y_max, self.y_min, num_cols)

        for row_index, x in enumerate(x_values):
            for col_index, y in enumerate(y_values):
                self.rob.go_to(x, y, z_plane[row_index, col_index])
                time.sleep(0.2 if col_index != 0 else 2)
                self.cam.capture_image(name=f"{folder}/tile_{tile}")
                tile += 1

        self.rob.end()
        self.scope.end()

    @staticmethod
    def get_folder(path):
        """Creates and returns a folder path."""
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created folder: {path}")
        return path

    def run(self, folder_path, z_points):
        """
        Runs the scanning process with the given folder path and z-height points.

        Parameters:
            folder_path (str): Path to the folder containing the data.
            z_points (list of tuples): List of (x, y, z) points for interpolating the focus plane.
        """
        try:
            folder = self.get_folder(folder_path)
            plane = self.interpolate_focus_plane(z_points)
            self.init_params()
            self.process_video(folder, z_plane=plane)
        except KeyboardInterrupt:
            print("Process interrupted by user.")
        finally:
            self.cam.release()


if __name__ == "__main__":
    scanner = TileScanner()
    scanner.run(folder_path=os.path.join(
        "content", "images", "df_c"), z_points=SystemConstants.FOCUS_PLANE_POINTS)
