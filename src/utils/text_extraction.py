import easyocr
import cv2
from PIL import Image, ImageDraw
import numpy as np

image_path = r"C:\Users\QATCH\Documents\GitHub\SensorQC\tile_0_0.jpg"
image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

resized = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

cv2.imwrite(
    r"C:\Users\QATCH\Documents\GitHub\SensorQC\tile_0_0.jpg",
    resized,
)
preprocessed_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB))

reader = easyocr.Reader(["en"])
result = reader.readtext(np.array(preprocessed_pil))
print(result)
draw = ImageDraw.Draw(preprocessed_pil)

for bbox, text, confidence in result:
    draw.polygon([tuple(point) for point in bbox], outline="red")
    draw.text((bbox[0][0], bbox[0][1]), text, fill="red")

preprocessed_pil.show()
