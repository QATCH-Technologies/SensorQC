import serial
import cv2
import time
import os
import subprocess
from image_stitcher import stitch

# Define boundaries and movement deltas
X_MIN, X_MAX = 0, 10
Y_MIN, Y_MAX = 0, 10
Z_FIXED = 10
FEED_RATE = 200
X_DELTA, Y_DELTA = 5, 5

ser = serial.Serial()
ser.port = "COM3"
ser.baudrate = 115200  # Set the appropriate baud rate
time.sleep(2)  # Wait for connection to establish


def control_machine(x, y):
    # Send G-code command to move the machine head to the specified coordinates
    gcode_command = f"G01 X{x:.2f} Y{y:.2f} Z{Z_FIXED:.2f} F{FEED_RATE:.2f}\n"
    print(gcode_command)
    ser.write(gcode_command.encode())
    time.sleep(0.5)  # Delay to allow movement to complete


def capture_image(frame, x, y, folder):
    # Save the image with the coordinates in the filename
    image_filename = os.path.join(folder, f"image_X{x:.2f}_Y{y:.2f}.jpg")
    cv2.imwrite(image_filename, frame)


def map_to_machine_axis(coordinate):
    # Map the camera coordinates to G-code workspace
    return coordinate * 0.1  # Example scaling factor (adjust as needed)


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

            # Capture an image
            ret, frame = cap.read()
            if ret:
                capture_image(frame, x, y, folder)
            else:
                print("Failed to capture image.")

            # Optional: Display the current position and captured image
            # cv2.imshow("Video Feed", frame)
            # cv2.waitKey(1)  # Brief pause to update display
    ser.close()
    cap.release()
    cv2.destroyAllWindows()


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
    stitch(input_folder, output_folder, 1, 1)
