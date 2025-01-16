import cv2
import time
import numpy as np
from dino_lite_edge import Camera, Microscope
from robot import Robot
from PIL import Image
from constants import SystemConstants
scope = Microscope()
cam = Camera(debug=False)
rob = Robot(debug=False)
rob.begin()
# rob.home()
rob.absolute_mode()
# scope.led_on(2)


def init_params():
    x, y, z, e = SystemConstants.INITIAL_POSITION
    rob.go_to(x, y, z)
    print("Camera has reached the initial position.")
    input("Enter to continue...")


def calculate_laplacian_variance(image):
    gray_image = cv2.cvtColor(
        image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()  # Compute the variance of the Laplacian
    return variance


def autofocus(z_range, step_size):
    z_min, z_max = z_range
    best_z = z_min
    best_frame = None
    max_sharpness = 0

    # Loop over Z values in the given range
    for z in np.arange(z_min, z_max, step_size):
        print(f"Moving to Z={z} for autofocus check...")
        rob.translate_z(z)  # Move to Z position via G-code

        # Capture a frame from the camera
        status, frame = cam._camera.read()
        if not status:
            print("Failed to capture image.")
            continue

        # Calculate sharpness using Laplacian variance
        sharpness = calculate_laplacian_variance(frame)
        print(f"Laplacian variance (sharpness) at Z={z}: {sharpness}")

        # If the current sharpness is higher than previous, update best Z
        if sharpness > max_sharpness:
            max_sharpness = sharpness
            best_z = z
            best_frame = frame

    print(f"Best Z-height for focus: {best_z}, Sharpness: {max_sharpness}")
    return best_z, best_frame


def calibrate_focus(corner_positions, z_range, step_size):
    z_heights = {}
    for corner, (x, y) in corner_positions.items():
        print(f"Moving to {corner}: (X={x}, Y={y})")
        rob.go_to(x, y, SystemConstants.Z_INITIAL)
        print(f"Running autofocus at {corner}...")
        z_height, b_frame = autofocus(z_range, step_size)
        cal_filename = f"cal_{corner}.jpg"
        cv2.imwrite(cal_filename, b_frame)
        print(f"Optimal Z-height at {corner}: {z_height}")
        z_heights[corner] = z_height
    return z_heights


if __name__ == "__main__":
    scope.led_on(state=1)
    init_params()
    z_height_results = calibrate_focus(
        SystemConstants.FOCUS_PLANE_POINTS, SystemConstants.FOCUS_RANGE, SystemConstants.FOCUS_STEP)
    print("Calibration Results (Z-heights at corners):")
    print(z_height_results)
    scope.end()
    rob.end()
