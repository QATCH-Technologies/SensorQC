import cv2
import time
import serial
import numpy as np
from dino_lite_edge import Camera, Microscope
from robot import Robot

Z_INITIAL = 9.0
Z_RANGE = (5.5, 6.5)
STEP_SIZE = 0.05
CORNERS = {
    "top_left": (108.2, 130.9),
    "top_right": (117.7, 128.4),
    "bottom_right": (117.2, 122.9),
    "bottom_left": (110.2, 122.4),
}
scope = Microscope()
cam = Camera(debug=False)
rob = Robot(debug=False)
rob.begin()
rob.absolute_mode()
scope.led_on(state=1)


def calculate_laplacian_variance(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = laplacian.var()  # Compute the variance of the Laplacian
    return variance


def autofocus(z_range, step_size):
    z_min, z_max = z_range
    best_z = z_min
    max_sharpness = 0

    # Loop over Z values in the given range
    for z in np.arange(z_min, z_max, step_size):
        print(f"Moving to Z={z} for autofocus check...")
        rob.translate_z(z)  # Move to Z position via G-code

        # Capture a frame from the camera
        status, frame = cam.__camera__.read()
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

    print(f"Best Z-height for focus: {best_z}, Sharpness: {max_sharpness}")
    return best_z


def calibrate_focus(corner_positions, z_range, step_size):
    z_heights = {}
    for corner, (x, y) in corner_positions.items():
        print(f"Moving to {corner}: (X={x}, Y={y})")
        rob.go_to(x, y, Z_INITIAL)
        print(f"Running autofocus at {corner}...")
        z_height = autofocus(z_range, step_size)
        print(f"Optimal Z-height at {corner}: {z_height}")
        z_heights[corner] = z_height
    return z_heights


if __name__ == "__main__":
    # Define the range of Z-values to explore
    # Z step size for autofocus
    z_height_results = calibrate_focus(CORNERS, Z_RANGE, STEP_SIZE)
    print("Calibration Results (Z-heights at corners):")
    print(z_height_results)
