import cv2
from ultralytics import YOLO

# Load the YOLOv9 model. Make sure you have trained it with your custom dataset.
model = YOLO("yolov9c.pt")


# Function to predict objects in an image using the chosen model.
def predict(chosen_model, img, classes=[], conf=0.5):
    if classes:
        results = chosen_model.predict(img, classes=classes, conf=conf)
    else:
        results = chosen_model.predict(img, conf=conf)

    return results


# Function to predict and detect objects, drawing bounding boxes and labels on the image.
def predict_and_detect(
    chosen_model, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1
):
    results = predict(chosen_model, img, classes, conf=conf)
    for result in results:
        for box in result.boxes:
            cv2.rectangle(
                img,
                (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                (255, 0, 0),  # Rectangle color (BGR format)
                rectangle_thickness,
            )
            # Display the class name above the bounding box.
            class_id = int(box.cls[0])  # Get class index
            class_name = (
                result.names[class_id] if class_id < len(result.names) else "Unknown"
            )
            cv2.putText(
                img,
                f"{class_name}",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),  # Text color (BGR format)
                text_thickness,
            )
    return img, results


# Read the image from the specified path.
image_path = r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\train\images\bottle_book.png"
image = cv2.imread(image_path)

# Perform prediction and detection on the image.
result_img, _ = predict_and_detect(model, image, classes=[], conf=0.5)

# Display the resulting image with bounding boxes and labels.
cv2.imshow("Image", result_img)
# Save the result image to the specified path.
cv2.imwrite(
    r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\train", result_img
)
cv2.waitKey(0)
cv2.destroyAllWindows()
