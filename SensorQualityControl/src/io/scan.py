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


# Define boundaries and movement deltas
INITIAL_POSITION = (104.70, 132.9, 8.80, 0.00)
FINAL_POSITION = (116.20, 122.40, 8.80, 0.00)
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


# scope = Microscope()
# cam = Camera()
# rob = Robot(debug=False)
# rob.begin()
# rob.absolute_mode()


def build_gradient_z(top_left, top_right, bottom_left, bottom_right):
    x_range = abs(X_MAX - X_MIN)
    y_range = abs(Y_MAX - Y_MIN)
    rows = int(x_range // X_DELTA)
    cols = int(y_range // -Y_DELTA)
    print(rows, cols)
    grid_1 = np.zeros((cols, rows))
    T = np.linspace(start=top_left, stop=bottom_left, num=cols)
    Q = np.linspace(start=top_right, stop=bottom_right, num=cols)
    grid_1[:, 0] = T
    grid_1[:, -1] = Q
    for i in range(cols):
        R = np.linspace(start=grid_1[i][0], stop=grid_1[i][-1], num=rows)
        grid_1[i] = R

    grid_2 = np.zeros((cols, rows))
    T = np.linspace(start=top_left, stop=top_right, num=rows)
    Q = np.linspace(start=bottom_left, stop=bottom_right, num=rows)
    grid_2[0,] = T
    grid_2[-1,] = Q
    for i in range(rows):
        R = np.linspace(start=grid_2[0][i], stop=grid_2[-1][i], num=cols)
        grid_2[:, i] = R

    print(np.equal(grid_1, grid_2))


def init_params():
    x, y, z, e = INITIAL_POSITION
    rob.go_to(x, y, z)

    # Wait for the print head to reach the initial position
    while True:
        response = rob.get_absolute_position()
        if (
            f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}" in response
        ):  # Adjust this condition based on your G-code machine's feedback
            break

    print("Camera has reached the initial position.")
    input("Enter to continue...")


def process_video(folder):
    init_params()
    tile = 1

    # Prepare a list to store tile locations
    tile_locations = []
    scope.disable_microtouch()
    scope.led_on(state=2)
    new_row = True
    for row_index, x in tqdm(
        enumerate(np.arange(X_MIN, X_MAX + X_DELTA, X_DELTA)), desc=">> Scanning"
    ):
        for col_index, y in enumerate(np.arange(Y_MAX, Y_MIN + -Y_DELTA, Y_DELTA)):
            rob.go_to(x, y, Z_FIXED)
            if new_row:
                time.sleep(2)
                new_row = False
            else:
                time.sleep(0.2)

            # for i in range(3):
            #     cam.__camera__.read()
            cam.capture_image(name=f"{folder}\\tile_{tile}")
            #
            tile += 1
        # for i in range(4):
        #     cam.__camera__.read()
        new_row = True
    # # Write the tile locations to a CSV file
    # csv_file_path = os.path.join(folder, "tile_locations.csv")
    # with open(csv_file_path, mode="w", newline="") as file:
    #     writer = csv.writer(file)
    #     writer.writerow(
    #         ["Tile Number", "Row Index", "Column Index"]
    #     )  # Write the header
    #     writer.writerows(tile_locations)  # Write the data
    rob.end()


def get_input_folder():
    # Define the folder path
    folder = os.path.join("content", "images", "raw_images")
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
    build_gradient_z(9.0, 8.8, 9.0, 8.7)
    # input_folder = get_input_folder()  # Get folder name from the user
    # output_folder = get_output_folder()  # Get folder name from the user
    # process_video(input_folder)  # Process video and capture images
