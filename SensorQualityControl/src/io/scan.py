import serial
import cv2
import time
import os
import signal
import sys
from image_stitcher import stitch
import easyocr
from PIL import Image
import numpy as np

# Define boundaries and movement deltas
BATCH_NAME = ""
SENSOR_HEIGHT = round(10.85)
SENSOR_WIDTH = round(11.35)
X_MIN = 0
X_MAX = X_MIN + SENSOR_WIDTH
Y_MIN = 0
Y_MAX = Y_MIN + SENSOR_HEIGHT
Z_FIXED = 10
FEED_RATE = 200
X_DELTA, Y_DELTA = 1, 1
SCALE_FACTOR = 1
ser = serial.Serial()
ser.port = "COM1"
ser.baudrate = 9600  # Set the appropriate baud rate
time.sleep(2)  # Wait for connection to establish


def signal_handler(sig, frame):
    print("Terminating the process...")
    ser.close()  # Close the serial port
    sys.exit(0)  # Exit the program


# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)


def control_machine(x, y):
    # Send G-code command to move the machine head to the specified coordinates
    gcode_command = f"G01 X{x:.2f} Y{y:.2f} Z{Z_FIXED:.2f} F{FEED_RATE:.2f}\n"
    print(gcode_command)
    ser.write(gcode_command.encode())
    ser.reset_output_buffer()
    time.sleep(0.5)  # Delay to allow movement to complete


def capture_image(frame, x, y, folder):
    # Save the image with the coordinates in the filename
    image_filename = os.path.join(folder, f"image_X{x:.2f}_Y{y:.2f}.jpg")
    cv2.imwrite(image_filename, frame)


def map_to_machine_axis(coordinate):
    # Map the camera coordinates to G-code workspace
    return coordinate * SCALE_FACTOR


def init_params():
    units_selection = f"G21"
    ser.write(units_selection.encode())
    positioning_absolute = f"G90"
    ser.write(positioning_absolute.encode())
    xy_plane = f"G17"
    ser.write(xy_plane.encode())


def process_video(folder):
    ser.open()
    # Open video feed from the camera
    cap = cv2.VideoCapture(0)
    init_params()
    # Loop through the defined box
    for x in range(X_MIN, X_MAX + 1, X_DELTA):
        for y in range(Y_MIN, Y_MAX + 1, Y_DELTA):
            # Move the machine
            gcode_x = map_to_machine_axis(x)
            gcode_y = map_to_machine_axis(y)
            control_machine(gcode_x, gcode_y)

            ret, frame = cap.read()
            if ret:
                capture_image(frame, x, y, folder)

                # # Display the live video feed
                # cv2.imshow("Live Video Feed", frame)

                # # Break the loop if 'q' is pressed
                # if cv2.waitKey(1) & 0xFF == ord("q"):
                #     break
            else:
                print("Failed to capture image.")

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
    stitch(input_folder, output_folder + "/stitched_image.jpg", 1, 1)
    # Load the image from the output folder
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
