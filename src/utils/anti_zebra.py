import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("20250130-112910-805.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect dark region using thresholding
# Adjust threshold value
_, mask = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

# Perform inpainting
inpainted = cv2.inpaint(image, mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)

# Display results
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(
    image, cv2.COLOR_BGR2RGB)), plt.title("Original Image")
plt.subplot(1, 3, 2), plt.imshow(mask, cmap='gray'), plt.title("Shadow Mask")
plt.subplot(1, 3, 3), plt.imshow(cv2.cvtColor(
    inpainted, cv2.COLOR_BGR2RGB)), plt.title("Corrected Image")
plt.show()
