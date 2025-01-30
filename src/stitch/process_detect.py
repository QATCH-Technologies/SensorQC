import logging
import multiprocessing
import numpy as np
import os
import cv2 as cv
import time

from logging.handlers import QueueHandler
from matplotlib import pyplot as plt
from process_common import CheckStd, SCALE_BY


class ProcessDetector(multiprocessing.Process):

    def __init__(self, queue_log, queue_rx1B, queue_tx2, queue_rx3,
                 log_level=logging.DEBUG):
        multiprocessing.Process.__init__(self)

        self.queue_log = queue_log
        self.log_level = log_level
        self.__log__()

        self.logger.debug("Detector process initializing...")

        self.queue_rx1B = queue_rx1B
        self.queue_tx2 = queue_tx2
        self.queue_rx3 = queue_rx3

        self.logger.debug("Detector process initialized")

    def __log__(self):

        self.logger = logging.getLogger(__class__.__name__)
        self.logger.addHandler(QueueHandler(self.queue_log))
        self.logger.setLevel(self.log_level)

    def config(self, image_path):
        """ Global configuration parameters common to all children processes """

        self.image_path = image_path

    def run(self):

        # Instantiate multiprocessing logger
        self.__log__()

        self.logger.debug("Starting...")
        # self.logger.warning('This is a warning')
        # self.logger.error('This is an error')

        self.offset_avg = None  # (0, 0)
        self.offset_std = None  # (0, 0)

        last_item_rxd = None
        sentinel_flags_rxd = 0

        while True:
            item = self.queue_rx1B.get()
            if item is None:  # Sentinel value to signal end of processing
                sentinel_flags_rxd += 1
                self.logger.debug(
                    f"Queue flag: {sentinel_flags_rxd} sentinel(s) received")
                self.queue_tx2.put(item)  # Forward sentinel to stitcher
                if sentinel_flags_rxd == 2:
                    break  # stop the process, as main and stitcher processes are finished
                else:
                    continue  # ignore 1st flag, and listen for more columns from stitcher

            self.logger.debug(f"Dequeueing: {item}")

            for i, tile in enumerate(item):

                tile = str(tile)  # force cast to string

                if not self.queue_rx3.empty():
                    resp = self.queue_rx3.get()
                    self.offset_avg, self.offset_std = resp
                    self.logger.debug(
                        f"AVG: {self.offset_avg}, STD: {self.offset_std}")

                if i > 0:
                    # self.logger.warning(f"last_tile: {last_tile}")
                    # self.logger.warning(f"this_tile: {this_tile}")
                    DEBUG_RANDOM_NOISE = False
                    if DEBUG_RANDOM_NOISE:
                        # Apply random error, and occasional craziness
                        from random import randint
                        time.sleep(1)
                        is_outlier = True if randint(0, 100) <= 25 else False
                        if is_outlier:
                            resp_x = randint(100, 300)
                            if randint(0, 1) == 0:
                                resp_x = -1 * resp_x
                            resp_y = randint(100, 300)
                            if randint(0, 1) == 0:
                                resp_y = -1 * resp_y
                        else:
                            resp_x = -45 + randint(-2, 2)
                            resp_y = 666 + randint(-20, 20)
                        resp = [(last_tile, tile), (resp_x, resp_y)]
                    else:
                        full_last_tile = os.path.join(
                            self.image_path, last_tile)
                        full_this_tile = os.path.join(self.image_path, tile)
                        is_column = False if tile.startswith("tile") else True
                        offset, anchor = self.compare_tiles(
                            full_last_tile, full_this_tile, is_column)
                        resp = [(last_tile, tile), offset, anchor]
                else:
                    # first tile in a row has no offset
                    # TODO: remove this and make stitcher not dependent upon it
                    resp = [(tile, tile), (0, 0), None]
                self.logger.debug(f"Offset: {resp}")
                self.queue_tx2.put(resp)
                last_tile = tile

            if last_item_rxd is not None and len(item) > 2:
                if len(last_item_rxd) == len(item):
                    for i in range(len(item)):
                        a_tile = str(last_item_rxd[i])
                        b_tile = str(item[i])
                        full_a_tile = os.path.join(self.image_path, a_tile)
                        full_b_tile = os.path.join(self.image_path, b_tile)
                        is_column = False if b_tile.startswith(
                            "tile") else True
                        offset, anchor = self.compare_tiles(
                            full_a_tile, full_b_tile, is_column)
                        resp = [(a_tile, b_tile), offset, anchor]
                        self.logger.debug(f"Column Offset: {resp}")
                        self.queue_tx2.put(resp)
                else:
                    self.logger.warning(
                        "Unable to check column alignments for these two rows."
                    )
                    self.logger.warning(
                        f"Reason: Length of this 'row' of tiles ({len(item)}) is " +
                        f"different from the prior one ({len(last_item_rxd)})."
                    )
            else:
                self.logger.debug(
                    "Skipping check of column alignments for this row.")

            if len(item) > 2:
                last_item_rxd = item

        self.logger.debug("Finished!")

    def compare_tiles(self, path1, path2, is_column=False):
        """ is_column: Set True to prevent return of 'smallest error' offset """

        img1 = cv.imread(path1, cv.IMREAD_GRAYSCALE)  # queryImage
        img2 = cv.imread(path2, cv.IMREAD_GRAYSCALE)  # trainImage

        if is_column:
            self.logger.info(f"Processing {os.path.basename(path2)}")

        scale = 0.1 if is_column else SCALE_BY
        img1 = cv.resize(img1, (0, 0), fx=scale, fy=scale)
        img2 = cv.resize(img2, (0, 0), fx=scale, fy=scale)

        DETECTOR_ORB = False
        DETECTOR_SIFT = False
        DETECTOR_FLANN = True

        SHOW_DEBUG_PLOTS = False

        if DETECTOR_ORB:

            # Initiate ORB detector
            orb = cv.ORB_create()

            # find the keypoints and descriptors with ORB
            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            # create BFMatcher object
            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

            # Match descriptors.
            matches = bf.match(des1, des2)

            # Sort them in the order of their distance.
            matches = sorted(matches, key=lambda x: x.distance)

            # Draw first 10 matches.
            img3 = cv.drawMatches(img1, kp1, img2, kp2, matches,
                                  None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            plt.imshow(img3), plt.show()

        if DETECTOR_SIFT:

            # Initiate SIFT detector
            sift = cv.SIFT_create()

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # BFMatcher with default params
            bf = cv.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])

            # cv.drawMatchesKnn expects list of lists as matches.
            img3 = cv.drawMatchesKnn(
                img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            plt.imshow(img3), plt.show()

        if DETECTOR_FLANN:

            # Initiate SIFT detector
            sift = cv.SIFT_create()

            # find the keypoints and descriptors with SIFT
            kp1, des1 = sift.detectAndCompute(img1, None)
            kp2, des2 = sift.detectAndCompute(img2, None)

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)  # or pass empty dictionary

            flann = cv.FlannBasedMatcher(index_params, search_params)

            try:
                matches = flann.knnMatch(des1, des2, k=2)
            except Exception as e:
                self.logger.error(f"Cannot align '{os.path.basename(path1)}' " +
                                  f"and '{os.path.basename(path2)}'", exc_info=1)
                return (-1, -1), {is_column: (-1, -1)}

            # Need to draw only good matches, so create a mask
            matchesMask = [[1, 0] for i in range(len(matches))]

            # ratio test as per Lowe's paper
            # matchScores = []
            img1_points, img2_points = [], []
            best_match_idx, best_match_ratio = -1, 1
            best_match_pt = (0, 0)
            best_match_offset = (0, 0)
            offset_errors = {}
            for i, (m, n) in enumerate(matches):
                this_ratio = m.distance / n.distance
                if this_ratio < best_match_ratio:
                    best_match_idx = i
                    best_match_ratio = this_ratio
                if self.offset_avg != None and is_column == False:  # and self.offset_std != None:
                    point1 = kp1[m.queryIdx].pt
                    point2 = kp2[m.trainIdx].pt
                    offset = (
                        int(point1[0] - point2[0]),
                        int(point1[1] - point2[1])
                    )
                    error_x = abs(self.offset_avg[0] - offset[0])
                    error_y = abs(self.offset_avg[1] - offset[1])
                    error = int(error_x + error_y)
                    # NOTE: this may overwrite duplicates with same 'error'
                    offset_errors[error] = offset
                # if m.distance < 0.25 * n.distance:
                    # point1 = kp1[m.queryIdx].pt
                    # point2 = kp2[m.trainIdx].pt
                    # img1_points.append(point1) #point1[0] - point2[0])
                    # img2_points.append(point2) #point1[1] - point2[1])
                    # matchesMask[i]=[1,0]
                    # print("Good match:", f"angle = {kp1[m.queryIdx].angle}")
                # else:
                #     matchScores.append(1.0)

            self.logger.debug(
                f"Best match: f{(best_match_idx, 1-best_match_ratio)}")
            if best_match_idx != -1:
                matchesMask[best_match_idx] = [1, 0]
                point1 = kp1[matches[best_match_idx][0].queryIdx].pt
                point2 = kp2[matches[best_match_idx][0].trainIdx].pt
                img1_points.append(point1)  # point1[0] - point2[0])
                img2_points.append(point2)  # point1[1] - point2[1])
                best_match_pt = (round(point1[0]), round(point1[1]))
                best_match_offset = (best_match_pt[0] - round(point2[0]),
                                     best_match_pt[1] - round(point2[0]))
            else:
                self.logger.error("No best match found! Please try again.")
                return (-1, -1), {is_column: (-1, -1)}

            # Limit number of key points to take the best ones
            # bestMatches = np.argsort(matchScores)[:10]
            # kp1 = tuple(np.array(kp1)[bestMatches])
            # kp2 = tuple(np.array(kp2)[bestMatches])
            # matches = tuple(np.array(matches)[bestMatches])
            # matchesMask = list(np.array(matchesMask)[bestMatches])

            x_offsets, y_offsets = [], []
            for i in range(len(img1_points)):
                x = int(img1_points[i][0])
                y = int(img1_points[i][1])
                # for _x in range(x-10, x+10):
                #     for _y in range(y-10, y+10):
                #         img3[_y][_x] = (0,255,0)
                # x = int(img2_points[i][0])
                # y = int(img2_points[i][1])
                # for _x in range(x-10, x+10):
                #     for _y in range(y-10, y+10):
                #         img3[_y][_x] = (0,255,0)

                x_offsets.append(img1_points[i][0] - img2_points[i][0])
                y_offsets.append(img1_points[i][1] - img2_points[i][1])

            debug_typical_x = -45
            # and abs(x_offsets[0] - debug_typical_x) > 5:

            x_offsets = np.sort(x_offsets)
            y_offsets = np.sort(y_offsets)

            if len(x_offsets) > 0:
                # self.logger.debug("Found offset")
                if is_column or True:
                    x_offsets = [x for x in x_offsets]
                    y_offsets = [y for y in y_offsets]

                # best_match_offset[0]
                middle_x = int(x_offsets[int(len(x_offsets) / 2)])

                # best_match_offset[1]
                middle_y = int(y_offsets[int(len(y_offsets) / 2)])

                valid_offset = True
                match_color = (0, 0, 0)  # black if no AVG set yet
                if self.offset_avg != None and self.offset_std != None and is_column == False:
                    check = CheckStd(
                        (middle_x, middle_y),
                        self.offset_avg,
                        self.offset_std
                    )
                    valid_offset = all(check)

                    # green if valid; otherwise blue
                    match_color = (0, 255, 0) if valid_offset else (0, 0, 255)

                if SHOW_DEBUG_PLOTS:
                    draw_params = dict(matchColor=(255, 0, 0),
                                       singlePointColor=(0, 0, 255),
                                       matchesMask=matchesMask,
                                       flags=cv.DrawMatchesFlags_DEFAULT)
                    img3 = cv.drawMatchesKnn(
                        img1, kp1, img2, kp2, matches, None, **draw_params)
                    cv.line(img3, best_match_pt,
                            (1280 + round(point2[0]), round(point2[1])),
                            match_color, 2)
                    plt.imshow(img3,), plt.show()

                if valid_offset:
                    # return best_match_offset, {is_column: best_match_pt}
                    return (middle_x, middle_y), {is_column: best_match_pt}
                # else, return lowest offset error, below
            else:
                self.logger.error(
                    "No KeyPoint offsets found, the array is empty!"
                )
                return (-1, -1), {is_column: (-1, -1)}

            # Return point with lowest offset error compared to AVG
            # ONLY do when the best point found fails the STD check
            if len(offset_errors) > 0:
                smallest_error = sorted(offset_errors.keys())[0]
                smallest_offset = offset_errors[smallest_error]
                self.logger.debug(f"Smallest error:  {smallest_error}")
                self.logger.debug(f"Smallest offset: {smallest_offset}")

                # TODO dev, removed this
                # return (-1, -1), {is_column: (-1, -1)}

                # return best, never do 'smallest'
                return (middle_x, middle_y), {is_column: best_match_pt}
                return smallest_offset, {is_column: best_match_pt}
            else:
                self.logger.error(
                    "No smallest error found, the array is empty!"
                )
                return (-1, -1), {is_column: (-1, -1)}

# TODO: Multithread offset calculations with stitching code to parallelize tasks and minimize execution time
# TODO: Use a running average and stddev() to detect outlier offset and use the average when errors observed
