# import the necessary packages
from imutils import paths
import numpy as np
import argparse
import imutils
import cv2
import os

# NOTE: Run with the following command format:
# py image_stitcher.py --images images/raw --output stitched.png --crop 1
#
# Source https://pyimagesearch.com/2018/12/17/image-stitching-with-opencv-and-python/


def stitch(input_path, output_path, crop=0, show=0):
    print("[INFO] loading images...")
    imagePaths = sorted(list(paths.list_images(input_path)))
    images = []
    for imagePath in imagePaths:
        print(imagePath)
        image = cv2.imread(imagePath)
        # cv2.imshow(f"{imagePath}", image)
        images.append(image)

    # print number of images found for stitching
    print("[DEBUG] found {} input images".format(len(images)))

    # initialize OpenCV's image stitcher object and then perform the image
    # stitching
    print("[INFO] stitching images...")
    stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
    # stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    (status, stitched) = stitcher.stitch(images)

    # if the status is '0', then OpenCV successfully performed image
    # stitching
    if status == 0:
        # check to see if we supposed to crop out the largest rectangular
        # region from the stitched image
        if crop > 0:
            # create a 10 pixel border surrounding the stitched image
            print("[INFO] cropping...")
            stitched = cv2.copyMakeBorder(
                stitched, 10, 10, 10, 10, cv2.BORDER_CONSTANT, (0, 0, 0)
            )

            # convert the stitched image to grayscale and threshold it
            # such that all pixels greater than zero are set to 255
            # (foreground) while all others remain 0 (background)
            gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]

            # find all external contours in the threshold image then find
            # the *largest* contour which will be the contour/outline of
            # the stitched image
            cnts = cv2.findContours(
                thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)

            # allocate memory for the mask which will contain the
            # rectangular bounding box of the stitched image region
            mask = np.zeros(thresh.shape, dtype="uint8")
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)

            # create two copies of the mask: one to serve as our actual
            # minimum rectangular region and another to serve as a counter
            # for how many pixels need to be removed to form the minimum
            # rectangular region
            minRect = mask.copy()
            sub = mask.copy()

            # keep looping until there are no non-zero pixels left in the
            # subtracted image
            while cv2.countNonZero(sub) > 0:
                # erode the minimum rectangular mask and then subtract
                # the thresholded image from the minimum rectangular mask
                # so we can count if there are any non-zero pixels left
                minRect = cv2.erode(minRect, None)
                sub = cv2.subtract(minRect, thresh)

            # find contours in the minimum rectangular mask and then
            # extract the bounding box (x, y)-coordinates
            cnts = cv2.findContours(
                minRect.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cnts = imutils.grab_contours(cnts)
            c = max(cnts, key=cv2.contourArea)
            (x, y, w, h) = cv2.boundingRect(c)

            # use the bounding box coordinates to extract the our final
            # stitched image
            stitched = stitched[y : y + h, x : x + w]

        # write the output stitched image to disk
        cv2.imwrite(output_path, stitched)

        # print useful success messages
        print("[INFO] image stitching successful")
        print(
            "[DEBUG] output created: {}".format(os.path.join(os.getcwd(), output_path))
        )

        # display the output stitched image to our screen (if desired)
        if show > 0:
            cv2.imshow("Stitched", stitched)
            cv2.waitKey(0)

    # otherwise the stitching failed, likely due to not enough keypoints)
    # being detected
    else:
        print("[INFO] image stitching failed ({})".format(status))
