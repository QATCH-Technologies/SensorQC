# Import required packages
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # your path may be different
)

# # Mention the installed location of Tesseract-OCR in your system
# pytesseract.pytesseract.tesseract_cmd = (
#     "/bin/tesseract"  # In case using colab after installing above modules
# )

# Read image from which text needs to be extracted
img = cv2.imread(
    r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\train\images\plant_book.png"
)

# Preprocessing the image starts

# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))

# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)

# Finding contours
contours, hierarchy = cv2.findContours(
    dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)

# Creating a copy of the image for contour drawing
contour_image = img.copy()

# A text file is created and flushed
file = open("recognized.txt", "w+")
file.write("")
file.close()

# Looping through the identified contours
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing the contour on the copied image
    cv2.drawContours(contour_image, [cnt], -1, (0, 255, 0), 2)

    # Cropping the text block for giving input to OCR
    cropped = img[y : y + h, x : x + w]

    # Open the file in append mode
    file = open("recognized.txt", "a")

    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(gray)
    # Display the image with contours
    cv2.imshow("Contoured Image", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # Appending the text into file
    file.write(text)
    file.write("\n")

    # Close the file
    file.close()
