import cv2
import numpy as np
from matplotlib import pyplot as plt


def gaussian_blending(overlap_region1, overlap_region2, overlap_width):
    x = np.linspace(-1, 1, overlap_width)
    gaussian = np.exp(-x**2 / 0.1)
    alpha = gaussian / gaussian.max()
    alpha = alpha.reshape(1, -1, 1)
    return (1 - alpha) * overlap_region1 + alpha * overlap_region2


def multiplicative_blending(overlap_region1, overlap_region2, overlap_width):
    alpha = np.linspace(0, 1, overlap_width).reshape(1, -1, 1)
    return np.sqrt((1 - alpha) * overlap_region1.astype(np.float32) * alpha * overlap_region2.astype(np.float32))


def harmonic_blending(overlap_region1, overlap_region2, overlap_width):
    """
    Harmonic mean blending: blends the overlap regions using their harmonic mean.
    This method reduces extreme values and creates a smooth transition.
    """
    return (2 * overlap_region1 * overlap_region2) / (overlap_region1 + overlap_region2 + 1e-6)


def seam_carving_blending(img1, img2, overlap_width):
    overlap_region1 = img1[:, -overlap_width:]
    overlap_region2 = img2[:, :overlap_width]

    gray1 = cv2.cvtColor(overlap_region1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(overlap_region2, cv2.COLOR_BGR2GRAY)
    energy_map1 = cv2.Sobel(gray1, cv2.CV_64F, 1, 0, ksize=3)**2 + \
        cv2.Sobel(gray1, cv2.CV_64F, 0, 1, ksize=3)**2
    energy_map2 = cv2.Sobel(gray2, cv2.CV_64F, 1, 0, ksize=3)**2 + \
        cv2.Sobel(gray2, cv2.CV_64F, 0, 1, ksize=3)**2
    combined_energy = energy_map1 + energy_map2
    seam_mask = np.zeros_like(combined_energy, dtype=np.uint8)
    for row in range(combined_energy.shape[0]):
        min_col = np.argmin(combined_energy[row])
        seam_mask[row, max(0, min_col - 1):min_col + 2] = 1

    alpha = seam_mask[:, :, np.newaxis].astype(np.float32)
    blended_region = (1 - alpha) * overlap_region1 + alpha * overlap_region2

    return np.hstack((img1[:, :-overlap_width], blended_region, img2[:, overlap_width:]))


def linear_blending(overlap_region1, overlap_region2, overlap_width):
    alpha = np.linspace(0, 1, overlap_width).reshape(1, -1, 1)
    return (1 - alpha) * overlap_region1 + alpha * overlap_region2


def smooth_transition_blending(img1, img2, overlap_width):

    height, _ = img1.shape[:2]
    alpha = np.tile(np.linspace(
        0, 1, overlap_width).reshape(1, -1), (height, 1))
    alpha = alpha[:, :, np.newaxis]

    overlap_region1 = img1[:, -overlap_width:]
    overlap_region2 = img2[:, :overlap_width]

    blended_region = (1 - alpha) * overlap_region1 + alpha * overlap_region2

    return np.hstack((img1[:, :-overlap_width], blended_region, img2[:, overlap_width:]))


def overlay_and_blend_images(image_path1, image_path2, overlap_width=200, blending_method='linear'):
    # Load images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    if img1 is None or img2 is None:
        raise ValueError("One or both image paths are invalid.")

    # Ensure both images have the same height
    if img1.shape[0] != img2.shape[0]:
        height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (img1.shape[1], height))
        img2 = cv2.resize(img2, (img2.shape[1], height))

    # Extract the overlapping regions
    overlap_region1 = img1[:, -overlap_width:]
    overlap_region2 = img2[:, :overlap_width]

    # Initialize blended region
    if blending_method == 'linear':
        blended_region = linear_blending(
            overlap_region1, overlap_region2, overlap_width)
        blended_image = np.hstack(
            (img1[:, :-overlap_width], blended_region, img2[:, overlap_width:]))
    elif blending_method == 'gaussian':
        blended_region = gaussian_blending(
            overlap_region1, overlap_region2, overlap_width)
        blended_image = np.hstack(
            (img1[:, :-overlap_width], blended_region, img2[:, overlap_width:]))
    elif blending_method == 'multiplicative':
        blended_region = multiplicative_blending(
            overlap_region1, overlap_region2, overlap_width)
        blended_image = np.hstack(
            (img1[:, :-overlap_width], blended_region, img2[:, overlap_width:]))
    elif blending_method == 'harmonic':
        blended_region = harmonic_blending(
            overlap_region1, overlap_region2, overlap_width)
        blended_image = np.hstack(
            (img1[:, :-overlap_width], blended_region, img2[:, overlap_width:]))
    elif blending_method == 'seam_carving':
        blended_image = seam_carving_blending(img1, img2, overlap_width)
    elif blending_method == 'smooth_transition':
        blended_image = smooth_transition_blending(img1, img2, overlap_width)
    else:
        raise ValueError(
            "Unsupported blending method. Use 'linear', 'gaussian', 'multiplicative', 'harmonic', or 'seam_carving'.")

    blended_image = blended_image.astype(np.uint8)
    return blended_image


# Example usage
if __name__ == "__main__":
    image_path1 = r"SensorQualityControl\af_Channel_1_Focus_Point.jpg"
    image_path2 = r"SensorQualityControl\af_Channel_2_Focus_Point.jpg"

    blending_methods = ['linear', 'gaussian',
                        'multiplicative', 'harmonic', 'seam_carving', "smooth_transition"]

    results = {}

    for method in blending_methods:
        results[method] = overlay_and_blend_images(
            image_path1, image_path2, overlap_width=200, blending_method=method)

    # Show results
    plt.figure(figsize=(20, 15))
    for i, method in enumerate(blending_methods, 1):
        plt.subplot(2, 3, i)
        plt.title(f"{method.capitalize()} Blending")
        plt.imshow(cv2.cvtColor(results[method], cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.show()

    # Optionally, save the blended images
    for method, result in results.items():
        cv2.imwrite(f"blended_{method}.jpg", result)
