import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from robot import Robot
from dino_lite_edge import Camera, Microscope
from scipy.spatial import Delaunay
from constants import (
    SystemConstants,
    RobotConstants,
    CameraConstants,
)
from tqdm import tqdm
import logging
import os
import cv2


logger = logging.getLogger(__name__)


class TileScanner:
    def __init__(
        self,
        robot: Robot,
        camera: Camera,
        microscope: Microscope,
        initial_position=SystemConstants.INITIAL_POSITION,
        final_position=SystemConstants.FINAL_POSITION,
        x_delta=SystemConstants.X_DELTA,
        y_delta=SystemConstants.Y_DELTA,
    ):
        logger.info("Initializing tile scanner.")
        self.initial_position = initial_position
        self.final_position = final_position
        self.x_min, self.y_max = initial_position.x, initial_position.y
        self.x_max, self.y_min = final_position.x, final_position.y
        self.x_delta = x_delta
        self.y_delta = y_delta
        self.scope = microscope
        self.cam = camera
        self.rob = robot

    def ceildiv(self, a: float, b: float) -> int:
        """Performs ceiling division."""
        return int(-(a // -b))

    def interpolate_focus_plane(self, z_points):
        logger.info(f"Interpolating autofocus plane using {len(z_points)} points.")
        # Extract x, y, z coordinates from the Position objects
        x_coords = np.array([point.x for point in z_points])
        y_coords = np.array([point.y for point in z_points])
        z_coords = np.array([point.z for point in z_points])

        # Generate the grid for the interpolation
        x_range = np.linspace(
            self.x_min,
            self.x_max,
            self.ceildiv(abs(self.x_max - self.x_min), self.x_delta),
        )
        y_range = np.linspace(
            self.y_min,
            self.y_max,
            self.ceildiv(abs(self.y_max - self.y_min), abs(self.y_delta)),
        )
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
        logger.info("Displaying interploated autofocus plane.")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(grid_x, grid_y, plane, cmap="viridis", edgecolor="k")
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-height")
        ax.set_title("Interpolated Focus Plane")
        plt.show()

    def init_params(self):
        self.rob.go_to(
            SystemConstants.INITIAL_POSITION.x,
            SystemConstants.INITIAL_POSITION.y,
            SystemConstants.INITIAL_POSITION.z,
        )
        logger.info("Gantry has reached the initial position.")
        input("Wait for gantry to stop moving. Press any key to proceed.")

    def report_column_frames(self, x, y, focus_plane, tile, row_index):
        """
        Captures each frame for a specific column.

        Parameters:
            x (float): The x-coordinate.
            y (float): The y-coordinate.
            focus_plane (ndarray): The focus plane data used for z-coordinate interpolation.
            tile (int): The current tile number.

        Returns:
            col_frames (list): A list of frames for the given column.
        """
        col_frames = []
        for col_index, y_value in enumerate(y):
            self.rob.go_to(x, y_value, focus_plane[row_index, col_index])
            time.sleep(
                RobotConstants.COLUMN_DELAY
                if col_index != 0
                else RobotConstants.ROW_DELAY
            )
            ret, frame = self.cam._camera.read()
            if not ret:
                logger.error("Failed to capture image from camera.")
                raise RuntimeError("Failed to capture image from camera.")
            col_frames.append(frame)
            tile += 1
        return col_frames

    def iterate_through_rows(self, x_values, y_values, focus_plane, output_dir):
        tile = 1
        num_rows, num_cols = focus_plane.shape
        total_tiles = num_rows * num_cols  # Total number of tiles

        # Initialize the progress bar
        with tqdm(total=total_tiles, desc="Capturing Tiles", unit="tile") as pbar:
            for row_index, x in enumerate(x_values):
                # Collect the column frames for this row
                col_frames = self.report_column_frames(
                    x, y_values, focus_plane, tile, row_index
                )

                # Create the directory if it doesn't exist
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                # Save each frame as a .jpg file
                for i, frame_data in enumerate(col_frames):
                    frame = frame_data
                    file_name = f"tile_{(row_index * num_cols) + i}.jpg"
                    file_path = os.path.join(output_dir, file_name)
                    cv2.imwrite(file_path, frame)  # Save the frame as a .jpg
                    logger.info(f"Saved {file_path}")

                tile += len(col_frames)
                pbar.update(num_cols)

    def run(self, z_points, runname):
        """
        Runs the scanning process with the given z-height points.

        Parameters:
            z_points (list of tuples): List of (x, y, z) points for interpolating the focus plane.
        """
        try:
            x_grid, y_grid, focus_plane = self.interpolate_focus_plane(z_points)
            self.plot_focus_plane(x_grid, y_grid, focus_plane)
            self.init_params()

            x_values = np.linspace(self.x_min, self.x_max, focus_plane.shape[0])
            y_values = np.linspace(self.y_max, self.y_min, focus_plane.shape[1])

            output_dir = os.path.join(
                r"C:\Users\QATCH\Documents\SVN Repos\SensorQC", runname
            )
            self.iterate_through_rows(x_values, y_values, focus_plane, output_dir)

        except KeyboardInterrupt:
            print("Process interrupted by user.")


if __name__ == "__main__":
    scanner = TileScanner()
    scanner.run(z_points=SystemConstants.FOCUS_PLANE_POINTS)
