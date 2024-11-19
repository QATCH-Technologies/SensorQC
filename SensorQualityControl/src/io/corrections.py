import numpy as np
import cv2
import os


def load_images_from_directory(directory, file_extension="jpg"):
    """
    Load all images with a given extension from a directory.

    Parameters:
    - directory (str): Path to the directory containing images.
    - file_extension (str): File extension to filter images (default: "jpg").

    Returns:
    - images (list of numpy arrays): List of loaded images.
    - filenames (list of str): List of corresponding filenames.
    """
    images = []
    filenames = []
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(file_extension):
            img_path = os.path.join(directory, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            if img is not None:
                images.append(img)
                filenames.append(filename)
    return images, filenames


def flat_field_correct_and_normalize_tiles(
    tile_dir, flat_field_path, output_dir, brightness_factor=2
):
    """
    Perform flat-field correction and global normalization on tiles in a directory.

    Parameters:
    - tile_dir (str): Directory containing tile images.
    - flat_field_path (str): Path to the flat-field image.
    - output_dir (str): Directory to save corrected and normalized tiles.
    """
    # Load tiles
    tiles, tile_names = load_images_from_directory(tile_dir)

    # Load the flat-field image
    flat_field = cv2.imread(flat_field_path, cv2.IMREAD_GRAYSCALE)
    if flat_field is None:
        raise FileNotFoundError(f"Flat-field image not found at: {flat_field_path}")

    # Avoid division by zero
    flat_field = np.where(flat_field == 0, 1, flat_field)

    corrected_tiles = []
    global_min, global_max = np.inf, -np.inf

    # Step 1: Perform flat-field correction and track global min and max values.
    for tile in tiles:
        # Flat-field correction
        corrected_tile = tile.astype(np.float32) / flat_field.astype(np.float32)
        corrected_tiles.append(corrected_tile)

        # Update global min and max
        global_min = min(global_min, corrected_tile.min())
        global_max = max(global_max, corrected_tile.max())

    # Step 2: Normalize all corrected tiles using global min and max.
    normalized_tiles = [
        ((tile - global_min) / (global_max - global_min) * 255).astype(np.uint8)
        for tile in corrected_tiles
    ]

    # Step 3: Apply brightness adjustment.
    brightened_tiles = [
        np.clip(tile * brightness_factor, 0, 255).astype(np.uint8)
        for tile in normalized_tiles
    ]

    # Step 4: Save brightened tiles to the output directory.
    os.makedirs(output_dir, exist_ok=True)
    for tile_name, brightened_tile in zip(tile_names, brightened_tiles):
        output_path = os.path.join(output_dir, tile_name)
        cv2.imwrite(output_path, brightened_tile)
        print(f"Saved brightened tile: {output_path}")


# Example Usage:
tile_directory = r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\images\bf_c_raw"  # Replace with the path to your tile images
flat_field_image = r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\flat_field_image.jpg"  # Replace with the path to your flat-field image
output_directory = r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\images\bf_c_cal_bright"  # Replace with the path to save corrected tiles

flat_field_correct_and_normalize_tiles(
    tile_directory, flat_field_image, output_directory
)
