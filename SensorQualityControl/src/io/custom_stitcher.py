from PIL import Image
import os
import math


def stitch_images_grid(
    image_paths, output_path, rows, cols, gap=30, bg_color=(255, 255, 255)
):
    """
    Stitch images together into a grid with specified rows and columns.

    Args:
        image_paths (list): List of file paths to images.
        output_path (str): Output path for the stitched image.
        rows (int): Number of vertical rows.
        cols (int): Number of images per row (columns).
        gap (int): Space between images in pixels.
        bg_color (tuple): Background color for the gap (default is white).

    Returns:
        None
    """
    if len(image_paths) != rows * cols:
        raise ValueError(
            f"Number of images ({len(image_paths)}) does not match grid size ({rows}x{cols})."
        )

    images = [Image.open(img) for img in image_paths]
    images = [img.rotate(270, expand=True) for img in images]
    # Find maximum width and height of images in the grid
    max_width = max(img.width for img in images)
    max_height = max(img.height for img in images)

    # Calculate dimensions of the final stitched image
    total_width = cols * max_width + (cols - 1) * gap
    total_height = rows * max_height + (rows - 1) * gap

    # Create a new blank image for the grid
    stitched_image = Image.new("RGB", (total_width, total_height), color=bg_color)

    # Paste each image into the grid
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x_offset = col * (max_width + gap)
        y_offset = row * (max_height + gap)
        stitched_image.paste(img, (x_offset, y_offset))

    # Save the stitched image
    stitched_image.save(output_path)
    print(f"Stitched image saved as {output_path}")


# Example usage
image_dir = (
    r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\images\raw_images"
)
output_file = "stitched_grid_image.jpg"

# Collect all image paths
image_files = [
    os.path.join(image_dir, img)
    for img in os.listdir(image_dir)
    if img.endswith(("png", "jpg", "jpeg"))
]

# Number of rows and columns in the grid
rows = 22  # For example, 3 rows
cols = 21  # For example, 4 images per row

# Make sure there are exactly rows * cols images
if len(image_files) < rows * cols:
    raise ValueError(f"Not enough images for a {rows}x{cols} grid.")

# Stitch images into a grid with gaps
stitch_images_grid(image_files[: rows * cols], output_file, rows, cols, gap=10)
