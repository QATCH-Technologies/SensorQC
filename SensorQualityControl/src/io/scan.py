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
INITIAL_POSITION = (105.20, 133.90, 8.80, 0.00)
FINAL_POSITION = (116.00, 121.90, 8.80, 0.00)
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
rob = Robot(debug=False)
rob.begin()
rob.absolute_mode()


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
    input_folder = get_input_folder()  # Get folder name from the user
    output_folder = get_output_folder()  # Get folder name from the user
    process_video(input_folder)  # Process video and capture images
