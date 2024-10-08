import cv2
import numpy as np


def highlight_overlapping_pixels(image1_path, image2_path):
    # Load the images in color
    img1 = cv2.imread(image1_path)
    img2 = cv2.imread(image2_path)

    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Convert images to binary (thresholding)
    _, binary1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
    _, binary2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)

    # Calculate overlapping pixels (intersection)
    overlap_mask = cv2.bitwise_and(binary1, binary2)

    # Highlight the overlapping pixels in red
    img1_highlighted = img1.copy()
    img2_highlighted = img2.copy()

    # Create a colored mask for the overlap (red color)
    red_color = [0, 0, 255]  # BGR format
    img1_highlighted[overlap_mask > 0] = red_color
    img2_highlighted[overlap_mask > 0] = red_color

    # Display the results
    cv2.imshow("Image 1 with Overlap", img1_highlighted)
    cv2.imshow("Image 2 with Overlap", img2_highlighted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage
image1_path = r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\images\raw_images\tile_29.jpg"
image2_path = r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\images\raw_images\tile_50.jpg"
highlight_overlapping_pixels(image1_path, image2_path)
