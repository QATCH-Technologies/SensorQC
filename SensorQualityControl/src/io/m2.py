import pandas as pd
from PIL import Image
import numpy as np


def stitch_tiles(csv_file, tile_dir, output_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Get the number of rows and columns for the final stitched image
    max_row = df["Row Index"].max() + 1
    max_col = df["Column Index"].max() + 1

    # Load a sample tile to get its dimensions
    sample_tile = Image.open(f"{tile_dir}/tile_1.jpg")
    tile_width, tile_height = sample_tile.size

    # Create a blank canvas for the final stitched image
    stitched_image = Image.new("RGB", (tile_width * max_col, tile_height * max_row))

    # Loop through each tile and place it in the correct position
    for index, row in df.iterrows():
        tile_number = row["Tile Number"]
        row_index = row["Row Index"]
        col_index = row["Column Index"]

        # Load the corresponding tile image
        tile_image = Image.open(f"{tile_dir}/tile_{tile_number}.jpg")

        # Calculate the position where the tile should be pasted
        x_position = row_index * tile_width
        y_position = col_index * tile_height

        # Paste the tile onto the final stitched image
        stitched_image.paste(tile_image, (x_position, y_position))

    # Save the stitched image
    stitched_image.save(output_file)
    print(f"Stitched image saved as {output_file}")


# Example usage
stitch_tiles(
    r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\images\raw_images\tile_locations.csv",
    r"C:\Users\QATCH\dev\SensorQC\SensorQualityControl\content\images\raw_images",
    "stitched_image.jpg",
)
