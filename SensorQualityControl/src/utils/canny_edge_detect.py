import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread(
    r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\train\images\plant_book.png",
    cv.IMREAD_GRAYSCALE,
)
assert img is not None, "file could not be read, check with os.path.exists()"
edges = cv.Canny(img, 100, 200)
cv.imwrite("edges.jpg", edges)

plt.subplot(121), plt.imshow(img, cmap="gray")
plt.title("Original Image"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap="gray")
plt.title("Edge Image"), plt.xticks([]), plt.yticks([])

plt.show()
