import json
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def display_coco_bboxes(coco_json_path, image_dir):
    """
    Display bounding boxes and categories on images from a COCO-formatted JSON file.

    Args:
        coco_json_path (str): Path to the COCO JSON annotation file.
        image_dir (str): Directory where images are stored.

    Returns:
        None
    """
    # Load COCO annotations
    with open(coco_json_path) as f:
        coco_data = json.load(f)

    # Map category IDs to category names
    category_mapping = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    # Create a dictionary of images for easy lookup by image ID
    images_info = {img["id"]: img for img in coco_data["images"]}

    # Create a dictionary of annotations grouped by image_id
    annotations_by_image = {}
    for ann in coco_data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)

    # Iterate through each image in the COCO dataset
    for image_id, image_info in images_info.items():
        # Load the image
        image_path = os.path.join(image_dir, image_info["file_name"])
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        # Get annotations for this image
        if image_id in annotations_by_image:
            for annotation in annotations_by_image[image_id]:
                bbox = annotation["bbox"]
                category_id = annotation["category_id"]
                category_name = category_mapping[category_id]

                # Bounding box coordinates: (x_min, y_min, width, height)
                x_min, y_min, width, height = bbox
                x_max = x_min + width
                y_max = y_min + height

                # Draw the bounding box (outline) in red
                draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

                # Add the category name at the top-left of the bounding box
                draw.text((x_min, y_min), category_name, fill="yellow")

        # Display the image with bounding boxes
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis("off")  # Hide axes
        plt.show()


# Example usage:
display_coco_bboxes(
    r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\train\annotations_coco.json",
    r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\train\images",
)
