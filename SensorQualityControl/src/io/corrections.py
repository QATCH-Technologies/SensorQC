from PIL import Image, ImageEnhance
import os
from glob import glob

# Directory containing images
image_directory = r'C:\Users\paulm\dev\SensorQC\SensorQualityControl\content\images\raw_images'


def gamma_correction(image, gamma=1.2):
    # Apply gamma correction
    inv_gamma = 1.0 / gamma
    lut = [int((i / 255.0) ** inv_gamma * 255) for i in range(256)]
    return image.point(lut * 3)


def adjust_brightness_contrast(image, brightness=1.2, contrast=1.2):
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(brightness)

    # Adjust contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(contrast)

    return image


def white_balance(image):
    # Simple white balance by equalizing the histogram
    image = ImageEnhance.Color(image).enhance(1.2)  # Slight color boost
    return image


def process_and_display_images(image_path):
    # Load the original image
    original_image = Image.open(image_path)

    # Apply preprocessing steps to create a modified version
    modified_image = adjust_brightness_contrast(original_image.copy())
    modified_image = gamma_correction(modified_image)
    modified_image = white_balance(modified_image)

    # Create a new blank image to combine both original and modified images side-by-side
    combined_width = original_image.width + modified_image.width
    combined_height = max(original_image.height, modified_image.height)
    combined_image = Image.new("RGB", (combined_width, combined_height))

    # Paste the original and modified images onto the combined image
    combined_image.paste(original_image, (0, 0))
    combined_image.paste(modified_image, (original_image.width, 0))

    # Display the combined image
    combined_image.show(title="Original vs Modified")
    print(f"Displayed original and modified image for {image_path}")

    # Wait for user input before proceeding to the next image
    input("Press Enter to proceed to the next image...")

    # Save the modified image back to the same path
    modified_image.save(image_path)


# Process each image in the directory
image_paths = glob(os.path.join(image_directory, '*.jpg')
                   )  # Adjust extension as needed

for image_path in image_paths:
    process_and_display_images(image_path)

print("All images in the directory have been processed, displayed, and saved.")
