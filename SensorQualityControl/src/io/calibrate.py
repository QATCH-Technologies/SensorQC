import cv2
import time
import numpy as np
from dino_lite_edge import Camera, Microscope
from robot import Robot
from PIL import Image

HOMING_TIME = 17
Z_INITIAL = 9.0
Z_RANGE = (3.5, 6.5)
STEP_SIZE = 0.05
CORNERS = {
    "top_left": (108.2, 130.9),
    "top_right": (117.7, 128.4),
    "bottom_right": (117.2, 122.9),
    "bottom_left": (110.2, 122.4),
}
INITIAL_POSITION = (108.2, 130.9, 6.19, 0.00)
scope = Microscope()
cam = Camera(debug=False)
rob = Robot(debug=False)
rob.begin()
rob.home()
rob.absolute_mode()


def generate_flat_field_image():
    """
    Generates a synthetic flat field image with uniform intensity.
    """
    scope.led_on(state=1)
    time.sleep(5)
    print("Capturing flat field images...")
    frame = cam.capture_image("flat_field_image", calibration=True)
    # Convert image to float for accurate division and avoid overflow
    image = frame.astype(np.float32)

    # Find the maximum pixel intensity
    max_intensity = np.max(image)

    # Avoid division by zero
    if max_intensity == 0:
        raise ValueError(
            "Image is completely dark; maximum intensity is zero.")

    # Scale image so that the brightest point becomes 1.0 (white)
    calibrated_image = image / max_intensity

    # Clip values to 1 to ensure they are in the range [0, 1]
    calibrated_image = np.clip(calibrated_image, 0, 1)

    # Optional: Convert back to 8-bit for standard image format, scaling up to 255
    calibrated_image = (calibrated_image * 255).astype(np.uint8)

    return calibrated_image


def generate_dark_field_image():
    """
    Generates a synthetic dark field image with uniform intensity.
    """
    print("Capturing flat field images...")
    scope.led_off()
    frame = cam.capture_image("dark_field_image", calibration=True)
    # Convert image to float for accurate division and avoid overflow
    image = frame.astype(np.float32)

    # Find the maximum pixel intensity
    max_intensity = np.max(image)

    # Avoid division by zero
    if max_intensity == 0:
        raise ValueError(
            "Image is completely dark; maximum intensity is zero.")

    # Scale image so that the brightest point becomes 1.0 (white)
    calibrated_image = image / max_intensity

    # Clip values to 1 to ensure they are in the range [0, 1]
    calibrated_image = np.clip(calibrated_image, 0, 1)

    # Optional: Convert back to 8-bit for standard image format, scaling up to 255
    calibrated_image = (calibrated_image * 255).astype(np.uint8)

    return calibrated_image


def init_params():
    x, y, z, e = INITIAL_POSITION
    rob.go_to(x, y, z)

    # # Wait for the print head to reach the initial position
    # while True:
    #     response = rob.get_absolute_position()
    #     if (
    #         f"X:{x:.2f} Y:{y:.2f} Z:{z:.2f}" in response
    #     ):  # Adjust this condition based on your G-code machine's feedback
    #         break
    # rob.absolute_mode()
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
    scope.led_off()

    ff_image = generate_flat_field_image()
    df_image = generate_dark_field_image()
    cv2.imwrite("flat_field_image.jpg", ff_image)
    cv2.imwrite("dark_field_image.jpg", df_image)
    scope.end()
    # init_params()
    # z_height_results = calibrate_focus(CORNERS, Z_RANGE, STEP_SIZE)
    # print("Calibration Results (Z-heights at corners):")
    # print(z_height_results)
    # scope.end()
    # rob.end()
