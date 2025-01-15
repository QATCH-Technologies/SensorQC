import serial
import cv2
import time
import os
import signal
import sys
import csv  # Import the CSV module
from image_stitcher import stitch
import easyocr
from PIL import Image
import numpy as np
from tqdm import tqdm
from robot import Robot
from dino_lite_edge import Camera, Microscope
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Z_INITIAL = 9.0
Z_RANGE = (5.5, 6.5)
STEP_SIZE = 0.05
# Define boundaries and movement deltas
INITIAL_POSITION = (108.4, 129.9, 4.65, 0.00)
FINAL_POSITION = (119.6, 119.4, 4.75, 0.00)
BATCH_NAME = ""
SENSOR_HEIGHT = 10.85
SENSOR_WIDTH = 11.35
X_MIN = INITIAL_POSITION[0]
X_MAX = FINAL_POSITION[0]
Y_MAX = INITIAL_POSITION[1]
Y_MIN = FINAL_POSITION[1]
Z_FIXED = INITIAL_POSITION[2]
FEED_RATE = 1000
X_DELTA, Y_DELTA = 0.5, -0.5
SCALE_FACTOR = 1


scope = Microscope()
cam = Camera()
rob = Robot(port="COM4", debug=False)
rob.begin()
rob.home()
rob.absolute_mode()
scope.disable_microtouch()
scope.led_on(state=1)


def ceildiv(a: float, b: float) -> int:
    return int(-(a // -b))


def interpolate_plane(top_left, top_right, bottom_left, bottom_right):
    x_range = abs(X_MAX - X_MIN)
    y_range = abs(Y_MAX - Y_MIN)
    rows = ceildiv(x_range, X_DELTA)
    cols = ceildiv(y_range, -Y_DELTA)
    plane = np.zeros((rows, cols))
    for i in range(rows):
        # Linear interpolation between top-left and bottom-left (for left column)
        z_left = top_left + (bottom_left - top_left) * (i / (rows - 1))

        # Linear interpolation between top-right and bottom-right (for right column)
        z_right = top_right + (bottom_right - top_right) * (i / (rows - 1))

        # Interpolate between the left and right sides for each row
        for j in range(cols):
            plane[i, j] = z_left + (z_right - z_left) * (j / (cols - 1))

    def plot_plane(plane):
        rows, cols = plane.shape
        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # Z-plane plot
        ax.plot_surface(X, Y, plane, cmap="viridis",
                        edgecolor="none", alpha=0.8)

        # Sensor plotting
        Z_flat = np.zeros_like(plane)
        ax.plot_surface(X, Y, Z_flat, color="gray",
                        edgecolor="none", alpha=0.5)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

    # plot_plane(plane)
    return plane


def init_params():
    # rob.home()
    scope.led_on(state=2)
    x, y, z, e = INITIAL_POSITION
    rob.go_to(x, y, z)
    scope.set_autoexposure(0)
    scope.set_exposure(414)

    # Wait for the print head to reach the initial position
    # while True:
    #     response = rob.get_absolute_position()
    #     if (
    #         f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}" in response
    #     ):  # Adjust this condition based on your G-code machine's feedback
    #         break
    rob.absolute_mode()
    print("Camera has reached the initial position.")
    input("Enter to continue...")


def process_video(folder, z_plane):
    init_params()
    tile = 1

    # Prepare a list to store tile locations
    tile_locations = []

    num_rows = ceildiv((X_MAX - X_MIN), X_DELTA)
    num_cols = ceildiv((Y_MAX - Y_MIN), Y_DELTA)
    total_tiles = num_rows * num_cols
    new_row = False
    num_rows, num_cols = z_plane.shape
    x_values = np.linspace(X_MIN, X_MAX, num_rows)
    y_values = np.linspace(Y_MAX, Y_MIN, num_cols)

    for row_index, x in enumerate(x_values):
        for col_index, y in enumerate(y_values):
            rob.go_to(x, y, z_plane[row_index, col_index])

            if new_row:
                time.sleep(2)
                new_row = False
            else:
                time.sleep(0.2)

            cam.capture_image(name=f"{folder}\\tile_{tile}")

            tile += 1
            # pbar.update(1)  # Update the progress bar for each tile processed
            new_row = True
    rob.end()
    scope.end()


def get_input_folder():
    # Define the folder path
    folder = os.path.join("content", "images", "df_c")
    if os.path.exists(folder):
        return folder
    else:
        os.makedirs(folder)  # Create the folder if it does not exist
        print(f"Created folder: {folder}")
    return folder


def get_output_folder():
    # Define the folder path
    folder = os.path.join("content", "images", "stitched_images")

    if not os.path.exists(folder):
        os.remove(folder)
        os.makedirs(folder)  # Create the folder if it does not exist
        print(f"Created folder: {folder}")
    else:
        print(f"Using existing folder: {folder}")
    return folder


if __name__ == "__main__":
    try:
        plane = interpolate_plane(4.90, 5.00, 4.90, 4.95)
        input_folder = get_input_folder()  # Get folder name from the user
        # output_folder = get_output_folder()  # Get folder name from the user
        print(f"Image path is: {input_folder}")
        # Process video and capture images
        process_video(input_folder, z_plane=plane)

    except KeyboardInterrupt:
        print("KeyboardInterrupt!")

    finally:
        cam.release()
