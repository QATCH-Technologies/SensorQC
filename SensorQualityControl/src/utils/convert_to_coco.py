import json
import os
from datetime import datetime


def labelme_to_coco(labelme_dir, output_file="annotations_coco.json"):
    """
    Convert Labelme JSON files in a directory to COCO format, with info, license, and supercategory sections.

    Args:
        labelme_dir (str): Directory containing Labelme annotation files.
        output_file (str): The file path for the output COCO JSON file.

    Returns:
        None
    """
    # Add info section to the COCO format
    coco = {
        "info": {
            "description": "Custom Dataset converted from Labelme format",
            "url": "http://example.com",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "Your Name or Organization",
            "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "licenses": [
            {
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License",
                "url": "http://creativecommons.org/licenses/by-nc-sa/4.0/",
            }
        ],
        "images": [],
        "annotations": [],
        "categories": [],
    }

    # Category ID mapping (Add your categories and supercategories here)
    category_mapping = {}
    supercategory_mapping = {}
    annotation_id = 1
    image_id = 1

    # Iterate over all JSON files in the Labelme directory
    for labelme_file in os.listdir(labelme_dir):
        if labelme_file.endswith(".json"):
            with open(os.path.join(labelme_dir, labelme_file)) as f:
                labelme_data = json.load(f)

            # Get image information
            file_name = labelme_data["imagePath"]
            image_width = labelme_data["imageWidth"]
            image_height = labelme_data["imageHeight"]

            # Add image metadata to COCO format
            coco["images"].append(
                {
                    "id": image_id,
                    "width": image_width,
                    "height": image_height,
                    "file_name": file_name,
                    "license": 1,  # Reference to the license ID
                }
            )

            # Add annotations for each object in the image
            for shape in labelme_data["shapes"]:
                label = shape["label"]
                supercategory = shape.get(
                    "supercategory", "default"
                )  # Get supercategory from the shape or default to 'default'
                points = shape["points"]  # List of points making up the polygon
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

                bbox_width = x_max - x_min
                bbox_height = y_max - y_min

                # Ensure the category and supercategory exist
                if label not in category_mapping:
                    category_id = len(category_mapping) + 1
                    category_mapping[label] = category_id

                    # Add supercategory if it doesn't exist
                    if supercategory not in supercategory_mapping:
                        supercategory_mapping[supercategory] = (
                            len(supercategory_mapping) + 1
                        )

                    coco["categories"].append(
                        {
                            "id": category_id,
                            "name": label,
                            "supercategory": supercategory,  # Assign supercategory
                        }
                    )

                # Add annotation in COCO format
                coco["annotations"].append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": category_mapping[label],
                        "bbox": [x_min, y_min, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "iscrowd": 0,
                    }
                )

                annotation_id += 1

            image_id += 1

    # Save the COCO-formatted annotations to a JSON file
    with open(output_file, "w") as out_file:
        json.dump(coco, out_file, indent=4)

    print(
        f"COCO annotations with info, license, and supercategory sections saved to {output_file}"
    )


# Example usage:
labelme_to_coco(
    r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\train\annotations",
    r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\train\annotations_coco.json",
)
