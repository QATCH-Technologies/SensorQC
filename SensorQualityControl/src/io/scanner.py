import time
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from robot import Robot
from dino_lite_edge import Camera, Microscope
from scipy.spatial import Delaunay
from constants import SystemConstants, RobotConstants, CameraConstants, MicroscopeConstants
from tqdm import tqdm


class TileScanner:
    def __init__(self, robot_port=RobotConstants.ROBOT_PORT, initial_position=SystemConstants.INITIAL_POSITION,
                 final_position=SystemConstants.FINAL_POSITION, x_delta=SystemConstants.X_DELTA, y_delta=SystemConstants.Y_DELTA):
        self.initial_position = initial_position
        self.final_position = final_position
        self.x_min, self.y_max = initial_position.x, initial_position.y
        self.x_max, self.y_min = final_position.x, final_position.y
        self.x_delta = x_delta
        self.y_delta = y_delta
        self.scope = Microscope(debug=True)
        self.cam = Camera(debug=True)
        self.rob = Robot(port=robot_port, debug=True)

    def ceildiv(self, a: float, b: float) -> int:
        """Performs ceiling division."""
        return int(-(a // -b))

    def interpolate_focus_plane(self, z_points):
        """
        Interpolates a plane based on any number of z-height points.

        Parameters:
            z_points (list of Position): List of Position objects defining the plane.

        Returns:
            np.ndarray: Interpolated plane as a 2D array.
        """
        # Extract x, y, z coordinates from the Position objects
        x_coords = np.array([point.x for point in z_points])
        y_coords = np.array([point.y for point in z_points])
        z_coords = np.array([point.z for point in z_points])

        # Generate the grid for the interpolation
        x_range = np.linspace(self.x_min, self.x_max, self.ceildiv(
            abs(self.x_max - self.x_min), self.x_delta))
        y_range = np.linspace(self.y_min, self.y_max, self.ceildiv(
            abs(self.y_max - self.y_min), abs(self.y_delta)))
        grid_x, grid_y = np.meshgrid(x_range, y_range)

        # Perform Delaunay triangulation
        tri = Delaunay(np.column_stack((x_coords, y_coords)))

        # Identify simplices (triangles) containing each grid point
        grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
        simplex = tri.find_simplex(grid_points)

        # Initialize plane array
        plane = np.full(grid_x.shape, np.nan)

        # Interpolate only where the simplex is valid
        valid = simplex >= 0
        for i, is_valid in enumerate(valid):
            if is_valid:
                simplex_idx = simplex[i]
                vertices = tri.simplices[simplex_idx]
                bary = tri.transform[simplex_idx, :2].dot(
                    grid_points[i] - tri.transform[simplex_idx, 2]
                )
                # Add the remaining barycentric coordinate
                bary = np.append(bary, 1 - bary.sum())
                plane.ravel()[i] = np.dot(bary, z_coords[vertices])

        # Replace NaNs with extrapolated values (optional)
        if np.isnan(plane).any():
            plane = np.nan_to_num(plane, nan=np.nanmean(plane))

        return grid_x, grid_y, plane

    def plot_focus_plane(self, grid_x, grid_y, plane):
        """
        Plots the interpolated focus plane in 3D.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(grid_x, grid_y, plane, cmap='viridis', edgecolor='k')
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-height')
        ax.set_title('Interpolated Focus Plane')
        plt.show()

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

    def process_video(self, z_plane):
        """Captures images at each tile location with a progress bar."""
        tile = 1
        num_rows, num_cols = z_plane.shape
        total_tiles = num_rows * num_cols  # Total number of tiles
        x_values = np.linspace(self.x_min, self.x_max, num_rows)
        y_values = np.linspace(self.y_max, self.y_min, num_cols)

        # Initialize the progress bar
        with tqdm(total=total_tiles, desc="Capturing Tiles", unit="tile") as pbar:
            for row_index, x in enumerate(x_values):
                col_frames = []
                for col_index, y in enumerate(y_values):
                    self.rob.go_to(x, y, z_plane[row_index, col_index])
                    time.sleep(RobotConstants.COLUMN_DELAY if col_index !=
                               0 else RobotConstants.ROW_DELAY)
                    ret, frame = self.cam._camera.read()
                    if not ret:
                        raise RuntimeError(
                            "Failed to capture image from camera.")
                    col_frames.append(
                        {"frame": frame, "location": f"tile_{tile}"})
                    tile += 1
                    pbar.update(1)
                yield col_frames
        self.rob.end()
        self.scope.end()

    def run(self, z_points):
        """
        Runs the scanning process with the given folder path and z-height points.

        Parameters:
            folder_path (str): Path to the folder containing the data.
            z_points (list of tuples): List of (x, y, z) points for interpolating the focus plane.
        """
        try:
            x_grid, y_grid, plane = self.interpolate_focus_plane(z_points)
            self.plot_focus_plane(x_grid, y_grid, plane)
            self.init_params()
            self.process_video(z_plane=plane)
        except KeyboardInterrupt:
            print("Process interrupted by user.")
        finally:
            self.cam.release()


if __name__ == "__main__":
    scanner = TileScanner()
    scanner.run(folder_path=os.path.join(
        "content", "images", "df_c"), z_points=SystemConstants.FOCUS_PLANE_POINTS)
