import cv2
import numpy as np


def find_reference_object(measurement_frame, reference_object, known_width=10.85, known_height=11.40):
    # Convert images to grayscale
    measurement_gray = cv2.cvtColor(measurement_frame, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_object, cv2.COLOR_BGR2GRAY)

    # Initialize ORB detector
    orb = cv2.ORB_create()

    # Detect keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(reference_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(measurement_gray, None)

    # Draw keypoints
    ref_keypoint_image = cv2.drawKeypoints(
        reference_gray, keypoints1, None, color=(0, 255, 0), flags=0)
    meas_keypoint_image = cv2.drawKeypoints(
        measurement_gray, keypoints2, None, color=(0, 255, 0), flags=0)

    cv2.imshow("Reference Object - ORB Keypoints", ref_keypoint_image)
    cv2.imshow("Measurement Frame - ORB Keypoints", meas_keypoint_image)

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 4:
        print("Not enough matches found!")
        return None

    # Draw matches
    match_img = cv2.drawMatches(
        reference_gray, keypoints1, measurement_gray, keypoints2, matches[:50], None, flags=2)
    cv2.imshow("Feature Matches", match_img)

    # Extract point correspondences
    src_pts = np.float32(
        [keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute homography matrix
    matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if matrix is None:
        print("Could not compute homography!")
        return None

    # Get bounding box of reference object in measurement frame
    h, w = reference_gray.shape
    ref_box = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    transformed_box = cv2.perspectiveTransform(ref_box, matrix)

    # Draw bounding box on measurement frame
    transformed_box = transformed_box.astype(int)
    meas_with_box = measurement_frame.copy()
    cv2.polylines(meas_with_box, [
                  transformed_box.reshape(-1, 2)], True, (0, 0, 255), 3)

    cv2.imshow("Detected Reference Object", meas_with_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Compute width and height in pixels
    transformed_w = np.linalg.norm(transformed_box[0] - transformed_box[1])
    transformed_h = np.linalg.norm(transformed_box[1] - transformed_box[2])

    # Compute pixel-to-mm ratio
    pixel_per_mm_w = transformed_w / known_width
    pixel_per_mm_h = transformed_h / known_height
    pixel_per_mm = (pixel_per_mm_w + pixel_per_mm_h) / 2

    # Get measurement frame dimensions
    img_height, img_width = measurement_gray.shape[:2]

    # Convert image size to mm
    frame_width_mm = img_width / pixel_per_mm
    frame_height_mm = img_height / pixel_per_mm

    return frame_width_mm, frame_height_mm


# Example usage
# Frame where you want to measure
measurement_frame = cv2.imread("reference.jpg")
# Image of reference object
reference_object = cv2.imread("tile_0_0.jpg")

frame_size = find_reference_object(measurement_frame, reference_object)
if frame_size:
    print(
        f"Estimated frame size: {frame_size[0]:.2f}mm x {frame_size[1]:.2f}mm")
