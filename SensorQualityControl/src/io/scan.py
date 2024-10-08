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

# Define boundaries and movement deltas
INITIAL_POSITION = (110.30, 131.80, 51.81, 0.00)
FINAL_POSITION = (121.30, 121.00, 51.81, 0.00)
BATCH_NAME = ""
SENSOR_HEIGHT = 10.85
SENSOR_WIDTH = 11.35
X_MIN = INITIAL_POSITION[0]
X_MAX = FINAL_POSITION[0]
Y_MAX = INITIAL_POSITION[1]
Y_MIN = FINAL_POSITION[1]
Z_FIXED = INITIAL_POSITION[2]
FEED_RATE = 200
X_DELTA, Y_DELTA = 0.5, -0.5
SCALE_FACTOR = 1
ser = serial.Serial()
ser.port = "COM4"
ser.baudrate = 115200  # Set the appropriate baud rate
time.sleep(2)  # Wait for connection to establish


# Final Position - X: -17, Y: -9, Z: -1
def signal_handler(sig, frame):
    print("Terminating the process...")
    ser.close()  # Close the serial port
    sys.exit(0)  # Exit the program


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def control_machine(x, y):
    # Send G-code command to move the machine head to the specified coordinates
    gcode_command = f"G01 X{x:.2f} Y{y:.2f} Z{Z_FIXED:.2f} F{FEED_RATE:.2f}\n"
    ser.write(gcode_command.encode())
    time.sleep(1)  # Delay to allow movement to complete


def capture_image(frame, tile_num, folder):
    # Save the image with the coordinates in the filename
    image_filename = os.path.join(folder, f"tile_{tile_num}.jpg")
    cv2.imwrite(image_filename, frame)


def map_to_machine_axis(coordinate):
    # Map the camera coordinates to G-code workspace
    return coordinate * SCALE_FACTOR


def init_params():
    units_selection = "G21"  # Set units to mm
    ser.write(units_selection.encode() + b"\n")

    positioning_absolute = "G90"  # Set positioning to absolute
    ser.write(positioning_absolute.encode() + b"\n")

    x, y, z, e = INITIAL_POSITION
    movement_command = f"G01 X{x} Y{y} Z{z} E{e}\n F{FEED_RATE:.2f}"
    ser.write(movement_command.encode())

    # Wait for the print head to reach the initial position
    while True:
        ser.write(b"M114\n")  # G-code for requesting the position
        time.sleep(0.5)

        response = ser.readline().decode("utf-8").strip()
        # Check for a response indicating the movement is complete
        if (
            f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}" in response
        ):  # Adjust this condition based on your G-code machine's feedback
            break

    print("Print head has reached the initial position.")
    input("Enter to continue..")


def process_video(folder):
    ser.open()
    # Open video feed from the camera
    cap = cv2.VideoCapture(1)
    init_params()
    tile = 1

    # Prepare a list to store tile locations
    tile_locations = []

    for row_index, x in tqdm(
        enumerate(np.arange(X_MIN, X_MAX + X_DELTA, X_DELTA)), desc=">> Scanning"
    ):
        for col_index, y in enumerate(np.arange(Y_MAX, Y_MIN + -Y_DELTA, Y_DELTA)):
            # Move the machine
            gcode_x = map_to_machine_axis(x)
            gcode_y = map_to_machine_axis(y)
            control_machine(gcode_x, gcode_y)

            ret, frame = cap.read()
            if ret:
                capture_image(frame, tile, folder)
                # Append the tile number, row index, and column index to the list
                tile_locations.append([tile, row_index, col_index])
                tile += 1
            else:
                print("Failed to capture image.")
        time.sleep(3)

    # Write the tile locations to a CSV file
    csv_file_path = os.path.join(folder, "tile_locations.csv")
    with open(csv_file_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Tile Number", "Row Index", "Column Index"]
        )  # Write the header
        writer.writerows(tile_locations)  # Write the data

    ser.close()
    cap.release()


def get_input_folder():
    # Define the folder path
    folder = os.path.join("content", "images", "raw_images")

    if not os.path.exists(folder):
        os.makedirs(folder)  # Create the folder if it does not exist
        print(f"Created folder: {folder}")
    else:
        print(f"Using existing folder: {folder}")
    return folder


def get_output_folder():
    # Define the folder path
    folder = os.path.join("content", "images", "stitched_images")

    if not os.path.exists(folder):
        os.makedirs(folder)  # Create the folder if it does not exist
        print(f"Created folder: {folder}")
    else:
        print(f"Using existing folder: {folder}")
    return folder


if __name__ == "__main__":
    input_folder = get_input_folder()  # Get folder name from the user
    output_folder = get_output_folder()  # Get folder name from the user
    process_video(input_folder)  # Process video and capture images
    from stitch2d import StructuredMosaic

    mosaic = StructuredMosaic(
        input_folder,
        dim=21,  # number of tiles in primary axis
        origin="upper left",  # position of first tile
        direction="vertical",  # primary axis (i.e., the direction to traverse first)
        pattern="raster",  # snake or raster
    )
    mosaic.align(limit=110)
    mosaic.build_out(from_placed=True)
    mosaic.show()

    image_path = os.path.join(
        output_folder, "stitched_image.jpg"
    )  # Assuming the image is saved as "stitched_image.jpg"
    image = cv2.imread(image_path)

    # Convert to grayscale and resize
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

    # Convert back to RGB for PIL
    preprocessed_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))

    # Perform OCR
    reader = easyocr.Reader(["en"])
    result = reader.readtext(np.array(preprocessed_pil))

    # Save the OCR result image
    ocr_output_path = os.path.join(
        output_folder, f"{BATCH_NAME}_result_{len(result)}.jpg"
    )
    preprocessed_pil.save(ocr_output_path)

    print(f"OCR result: {result}")
    print(f"OCR result image saved at: {ocr_output_path}")
