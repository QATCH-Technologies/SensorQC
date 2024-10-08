import os
from stitching import Stitcher
import cv2

stitcher = Stitcher()


def load_images_from_directory(directory):
    # Load all images from the specified directory using OpenCV
    image_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]
    images = [cv2.imread(img) for img in image_files if cv2.imread(img) is not None]
    rotated_images = [
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) for image in images
    ]  # Apply rotation
    rotated_images = [
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) for image in rotated_images
    ]  # 180 degrees
    rotated_images = [
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE) for image in rotated_images
    ]  # 270 degrees

    return rotated_images


def stitch_images_from_directory(directory):
    images = load_images_from_directory(directory)

    # Ensure there are enough images to stitch
    if len(images) < 2:
        raise ValueError("Need at least two images to perform stitching")

    # Stitch images together using the stitching package
    stitched_image = stitcher.stitch(images)
    return stitched_image


# Define the directory containing raw images
raw_images_directory = (
    r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\images\stitched_images"
)

# Stitch the images
stitched_image = stitch_images_from_directory(raw_images_directory)

# Save the stitched result
stitched_image.save("stitched_output.jpg")

print("Stitching completed and saved as 'stitched_output.jpg'")
