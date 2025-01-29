from process_common import CheckStd, NUM_ROWS, SCALE_BY, CORRECT_DF
from matplotlib import pyplot as plt
from logging.handlers import QueueHandler
from enum import Enum
import cv2 as cv
import logging
import multiprocessing
import numpy as np
import os


class BlendType(Enum):
    NONE = None
    ALPHA_50_50 = "alpha-50-50"
    LINEAR_BLEND = "linear-blend"


CV_IO_MAX_IMAGE_PIXELS = pow(2, 30)
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(CV_IO_MAX_IMAGE_PIXELS)

BLEND_OVERLAP = BlendType.LINEAR_BLEND
COLOR_WHITE = (255, 255, 255)
COLOR_PINK = (255, 0, 128)


class ProcessStitcher(multiprocessing.Process):

    def __init__(self, queue_log, queue_rx1A, queue_rx1B, queue_rx2, queue_tx3,
                 log_level=logging.DEBUG, trans_color=COLOR_WHITE):  # neon pink transparent color
        multiprocessing.Process.__init__(self)

        self.queue_log = queue_log
        self.log_level = log_level
        self.__log__()

        self.logger.debug("Stitcher process initializing...")

        self.queue_rx1A = queue_rx1A
        self.queue_rx1B = queue_rx1B  # allows stitch to add col images to detect queue
        self.queue_rx2 = queue_rx2
        self.queue_tx3 = queue_tx3

        self.trans_color = trans_color

        self.logger.debug("Stitcher process initialized")

    def __log__(self):

        self.logger = logging.getLogger(__class__.__name__)
        if self.queue_log is not None:
            self.logger.addHandler(QueueHandler(self.queue_log))
        else:
            h = logging.StreamHandler()
            f = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            h.setFormatter(f)
            self.logger.addHandler(h)
        self.logger.setLevel(self.log_level)

    def config(self, image_path):
        """ Global configuration parameters common to all children processes """

        self.image_path = image_path

    def overlay_images(self, bg, fg, x_offset, y_offset, is_tile=False, is_column=False):
        """Overlays fg on bg with 50% transparency at (x_offset, y_offset)."""

        # Ensure images have the same number of channels
        if bg.shape[2] != fg.shape[2]:
            raise ValueError("Images must have the same number of channels")

        # Adjust offsets for positive absolute reference
        # y_offset = y_offset + fg.shape[0]
        # x_offset = x_offset + fg.shape[1]

        # print(x_offset, y_offset)

        # Extend image with pink background
        top1, bottom1 = max(0, -y_offset), max(0, y_offset)
        left1, right1 = max(0, -x_offset), max(0, x_offset)

        # print(top1, bottom1, left1, right1)
        out1 = cv.copyMakeBorder(bg,
                                 top1, bottom1, left1, right1,
                                 cv.BORDER_CONSTANT,
                                 value=self.trans_color)

        # Create a region of interest (ROI) for the overlap region
        y1, y2 = max(0, y_offset) + \
            (bg.shape[0] - fg.shape[0]), out1.shape[0] - abs(y_offset)
        x1, x2 = max(0, x_offset) + \
            (bg.shape[1] - fg.shape[1]), out1.shape[1] - abs(x_offset)

        if y_offset < 0:
            # offset y1/y2 to zero y1
            y2 -= y1
            y1 -= y1
        if x_offset < 0:
            # offset x1/x2 to zero x1
            x2 -= x1
            x1 -= x1

        # Overlap background and foreground images,
        # with no alpha in overlap region (yet)
        out = out1.copy()
        out[y1:y1 + fg.shape[0], x1:x1 + fg.shape[1]] = fg

        # Extend image with pink background
        top2, bottom2 = out.shape[0] - \
            (fg.shape[0] - y_offset) - bottom1, max(0, -y_offset)
        left2, right2 = out.shape[1] - \
            (fg.shape[1] - x_offset) - right1, max(0, -x_offset)

        if y_offset < 0:
            bottom2 += top2
            top2 = 0
        if x_offset < 0:
            right2 += left2
            left2 = 0

        # print(top2, bottom2, left2, right2)
        out2 = cv.copyMakeBorder(fg,
                                 top2, bottom2, left2, right2,
                                 cv.BORDER_CONSTANT,
                                 value=self.trans_color)

        # self.plot_image(out1)
        # self.plot_image(out2)
        # self.plot_image(out)

        if BLEND_OVERLAP == BlendType.NONE:
            return out

        try:
            # Blend the foreground image onto the background image
            bg_roi = out1[y1+top1:y2, x1+left1:x2]
            fg_roi = fg[max(0, -y_offset):fg.shape[0]-max(0, y_offset),
                        max(0, -x_offset):fg.shape[1]-max(0, x_offset)]

            if BLEND_OVERLAP == BlendType.ALPHA_50_50:
                alpha = 0.5  # Transparency level
                blended_roi = cv.addWeighted(bg_roi, alpha, fg_roi, alpha, 0)

            elif BLEND_OVERLAP == BlendType.LINEAR_BLEND:
                if is_tile:
                    roi_h = bg_roi.shape[:2][0]
                    alpha_h = np.linspace(0, 1, roi_h).reshape(-1, 1, 1)
                    blended_roi = (1 - alpha_h) * bg_roi + alpha_h * fg_roi
                elif is_column:
                    roi_w = bg_roi.shape[:2][1]
                    alpha_w = np.linspace(0, 1, roi_w).reshape(1, -1, 1)
                    blended_roi = (1 - alpha_w) * bg_roi + alpha_w * fg_roi
                else:
                    alpha = 0.5  # Transparency level
                    blended_roi = cv.addWeighted(
                        bg_roi, alpha, fg_roi, alpha, 0)

            else:
                raise ValueError(f"Unknown BlendType: {BLEND_OVERLAP}")

            # self.plot_image(blended_roi)

            # print(bg_roi.shape, fg_roi.shape)
            # self.plot_image(bg_roi)
            # self.plot_image(fg_roi)

            # # Create a mask for the specified color
            # bg_mask = np.all(bg_roi == self.trans_color, axis=-1)
            # fg_mask = np.all(fg_roi == self.trans_color, axis=-1)

            # # Do not overlay transparent areas of background onto foreground
            # if np.any(bg_mask):
            #     print("bg_count", np.count_nonzero(bg_mask.flatten()))
            #     blended_roi[bg_mask] = (0, 0, 0) # fg_roi[bg_mask, 0] # red
            #     # blended_roi[bg_mask, 1] = fg_roi[bg_mask, 1] # green
            #     # blended_roi[bg_mask, 2] = fg_roi[bg_mask, 2] # blue

            # # Do not overlay transparent areas of foreground onto background
            # if np.any(fg_mask):
            #     print("fg_count", np.count_nonzero(fg_mask.flatten()))
            #     blended_roi[fg_mask] = bg_roi[fg_mask] # red
            #     # blended_roi[fg_mask, 1] = bg_roi[fg_mask, 1] # green
            #     # blended_roi[fg_mask, 2] = bg_roi[fg_mask, 2] # blue

            lowerb = list(self.trans_color[::-1])
            upperb = lowerb.copy()
            max_jpeg_compression_delta = 5
            for i in range(len(lowerb)):
                lowerb[i] -= max_jpeg_compression_delta
                upperb[i] += max_jpeg_compression_delta
            lowerb = tuple(lowerb)
            upperb = tuple(upperb)
            bg_mask = np.array(cv.inRange(bg_roi, lowerb, upperb), dtype=bool)
            fg_mask = np.array(cv.inRange(fg_roi, lowerb, upperb), dtype=bool)
            blended_roi[bg_mask] = fg_roi[bg_mask]
            blended_roi[fg_mask] = bg_roi[fg_mask]

            # pixel_total = len(blended_roi.flatten())
            # pixel_count = 0
            # t_color = list(self.trans_color[::-1])
            # for pixel_Y in range(len(blended_roi)):
            #     for pixel_X in range(len(blended_roi[pixel_Y])):
            #         pixel_count += 1
            #         print(f"Processing pixel {pixel_count} / {pixel_total}")
            #         bg_color = list(bg_roi[pixel_Y][pixel_X])
            #         fg_color = list(fg_roi[pixel_Y][pixel_X])
            #         if bg_color == t_color:
            #             blended_roi[pixel_Y][pixel_X] = (0, 0, 0) # fg_roi[pixel]
            #         if fg_color == t_color:
            #             blended_roi[pixel_Y][pixel_X] = (255, 255, 255) # bg_roi[pixel]

            # self.plot_image(blended_roi)

            out[y1+top1:y2, x1+left1:x2] = blended_roi

        except Exception as e:
            self.logger.error("Error while overlaping tiles:")
            self.logger.exception(e)

            self.plot_image(out1)
            self.plot_image(out2)
            self.plot_image(out)
            input("Press Enter to continue...")

        return out

    def rotate_image(self, image):
        self.logger.info("Rotating image")

        # prep for finding transparent color border in image
        lowerb = list(self.trans_color)
        upperb = lowerb.copy()
        max_jpeg_compression_delta = 5
        for i in range(len(lowerb)):
            lowerb[i] -= max_jpeg_compression_delta
            upperb[i] += max_jpeg_compression_delta
        lowerb = tuple(lowerb)
        upperb = tuple(upperb)

        bg_mask = np.array(cv.inRange(image, lowerb, upperb), dtype=bool)

        # grab the dimensions of the image and calculate the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        def get_first_true_index(lst):
            for i, item in enumerate(lst):
                if item:
                    return i
            return -1  # Return -1 if no True value is found

        col_with_no_border = 0
        for i in range(len(bg_mask)):
            pixel_line = bg_mask[i]
            val = get_first_true_index(pixel_line)
            if val >= 0:
                col_with_no_border = val
                break

        # rotate our image by +/- 1 degrees around the center of the image
        R = 1 if col_with_no_border < cX else -1  # CCW = 1 deg, CW = -1 deg
        M = cv.getRotationMatrix2D((cX, cY), R, 1.0)

        min_border_pixels = np.max(bg_mask.shape)
        max_border_pixels = 0
        rotated = image.copy()
        for deg in range(90):  # max rotation 90 degrees before giving up
            test_image = cv.warpAffine(
                rotated, M, (w, h), borderMode=cv.BORDER_CONSTANT, borderValue=self.trans_color)
            bg_mask = np.array(cv.inRange(
                test_image, lowerb, upperb), dtype=bool)

            border_pixels_to_crop = 0
            ALLOW_SAWTOOTH_EDGES = False
            for i in range(len(bg_mask)):
                pixel_line = bg_mask[i]
                pixel_line = pixel_line[i:len(pixel_line)-i]
                if ALLOW_SAWTOOTH_EDGES:
                    if not np.all(pixel_line):
                        break
                else:
                    if not np.any(pixel_line):
                        break
                border_pixels_to_crop += 1

            is_new_min, is_new_max = False, False
            if border_pixels_to_crop >= max_border_pixels:
                max_border_pixels = border_pixels_to_crop
                is_new_max = True
            if border_pixels_to_crop <= min_border_pixels:
                min_border_pixels = border_pixels_to_crop
                is_new_min = True

            time_to_stop = False
            if ALLOW_SAWTOOTH_EDGES:
                if not is_new_max:
                    time_to_stop = True
            else:
                if not is_new_min:
                    time_to_stop = True

            if not time_to_stop:
                rotated = test_image
            else:
                self.logger.debug(f"Rotated {R*deg} deg")
                break

        # crop the rotated image to remove the pink border
        top_border, bottom_border = border_pixels_to_crop, -border_pixels_to_crop
        left_border, right_border = border_pixels_to_crop, -border_pixels_to_crop
        rotated = rotated[top_border:bottom_border, left_border:right_border]

        if ALLOW_SAWTOOTH_EDGES:
            bg_mask = np.array(cv.inRange(rotated, lowerb, upperb), dtype=bool)
            rotated[bg_mask] = (0, 0, 0)  # (255, 255, 255)

        return rotated

    def plot_image(self, img, figsize_in_inches=(5, 5)):
        fig, ax = plt.subplots(figsize=figsize_in_inches)
        ax.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
        plt.show()

    def df_prepare(self, shape):
        if not CORRECT_DF:
            return

        field_path = os.path.join(
            r"C:\Users\Alexander J. Ross\Documents\SVN Repos\SensorQC")
        vig1 = cv.imread(os.path.join(field_path, 'flatfield_correction_df.jpg'),
                         cv.IMREAD_GRAYSCALE)  # Read vignette template as grayscale
        vig2 = cv.imread(os.path.join(field_path, 'darkfield_correction_df.jpg'),
                         cv.IMREAD_GRAYSCALE)  # Read vignette template as grayscale

        alpha = 0.25  # Transparency level
        vig = cv.addWeighted(vig1, alpha, vig2, 1 - alpha, 0)

        # Resize the image
        vig = cv.resize(vig, shape)

        # Apply median filter for removing artifacts and extreem pixels.
        vig = cv.medianBlur(vig, 15)

        # Convert vig to float32 in range [0, 1]
        vig_norm = vig.astype(np.float32) / 255
        # Blur the vignette template (because there are still artifacts, maybe because SO convered the image to JPEG).
        vig_norm = cv.GaussianBlur(vig_norm, (51, 51), 30)
        # vig_max_val = vig_norm.max()  # For avoiding "false colors" we may use the maximum instead of the mean.
        vig_mean_val = cv.mean(vig_norm)[0]
        # vig_max_val / vig_norm
        inv_norm_vig = vig_mean_val / vig_norm  # Compute G = m/F

        # Convert inv_norm_vig to 3 channels before using cv2.multiply. https://stackoverflow.com/a/48338932/4926757
        self.df_normalizer = cv.cvtColor(inv_norm_vig, cv.COLOR_GRAY2BGR)

    def df_correct(self, image):
        if not CORRECT_DF:
            return image

        # Compute: C = R * G
        return cv.multiply(image, self.df_normalizer, dtype=cv.CV_8U)

    def place_tiles_absolutely(self):
        tile_pos_file = os.path.join(self.image_path, "tile_positions.csv")
        if os.path.exists(tile_pos_file):
            absolute_positions = {}
            with open(tile_pos_file, "r") as f:
                for line in f.readlines():
                    tile = line[0:line.find(':')]
                    pos_x = line[line.find('(') + 1:line.find(',')]
                    pos_y = line[line.find(',') + 1:line.find(')')]
                    absolute_positions[tile] = (int(pos_x), int(pos_y))
            IS_TILE_ZERO_INDEXED = True if str(
                list(absolute_positions.keys())[0]).find("0") else False
            total_tiles_queued = len(absolute_positions.keys())
            num_tiles_per_row = NUM_ROWS
            grid_rows = num_tiles_per_row
            grid_cols = int(total_tiles_queued / num_tiles_per_row)
            assert grid_cols % 1 == 0, "Number of tiles not divisible by 'rows'. Check and try again."
            final_image = None
            for y in range(grid_cols):
                tiles_this_row = []
                for x in range(y*grid_rows, (y+1)*grid_rows):
                    if IS_TILE_ZERO_INDEXED:
                        tile = f"tile_{x}.jpg"
                    else:
                        tile = f"tile_{x+1}.jpg"
                    tiles_this_row.append(tile)
                # self.logger.debug(tiles_this_row)
                last_pos = (0, 0)
                for i, tile in enumerate(tiles_this_row):
                    this_tile = os.path.join(self.image_path, tile)
                    if i == 0:
                        overlay_image = cv.resize(
                            cv.imread(this_tile), (0, 0), fx=SCALE_BY, fy=SCALE_BY)
                        tile_shape = tuple(overlay_image.shape[0:2][::-1])
                        self.df_prepare(tile_shape)
                        overlay_image = self.df_correct(overlay_image)
                    else:
                        overlay_shape = tuple(overlay_image.shape[0:2][::-1])
                        this_position = absolute_positions[tile]
                        offset_x = this_position[0] - last_pos[0]
                        offset_y = this_position[1] - last_pos[1]
                        overlay_image = self.overlay_images(
                            overlay_image,
                            self.df_correct(cv.resize(cv.imread(this_tile), (0, 0),
                                                      fx=SCALE_BY, fy=SCALE_BY)),
                            offset_x,
                            offset_y,
                            is_tile=True
                        )
                    last_pos = absolute_positions[tile]
                    # self.plot_image(overlay_image, (20, 20))
                self.logger.info(f"Stitching column {y+1}")
                if y == 0:
                    final_image = overlay_image
                else:
                    # absolute_positions[tiles_this_row[-1]][0] - absolute_positions[tiles_this_row[0]][0]
                    horizontal_offset = 0
                    # absolute_positions[tiles_this_row[-1]][1] - absolute_positions[tiles_this_row[0]][1]
                    vertical_offset = 0
                    # last_tile = f"tile_{x+1-22-21}.jpg"
                    this_position = absolute_positions[tiles_this_row[0]]
                    offset_x = this_position[0] - \
                        last_col_pos[0] - 2*horizontal_offset
                    offset_y = this_position[1] - \
                        last_col_pos[1] + vertical_offset
                    final_image = self.overlay_images(
                        final_image,
                        overlay_image,
                        offset_x,
                        offset_y,
                        is_column=True
                    )
                last_col_pos = absolute_positions[tiles_this_row[0]]
                # self.plot_image(final_image, (20, 20))
                # out_path = os.path.join(self.image_path, f"column_{y+1}.jpg")
                # cv.imwrite(out_path, overlay_image)

            # Rotate image by +/-1 deg at a time until pink edges are gone
            final_image = self.rotate_image(final_image)

            out_path = os.path.join(self.image_path, f"stitched_absolute.jpg")
            self.logger.info(
                f"Saving final image: {os.path.basename(out_path)}")
            # os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv.imwrite(out_path, final_image)
            # self.plot_image(final_image, (20, 20))
            # self.logger.info(f"Finished")
            # input()

    def create_stitched_matrix(self):
        tile_pos_file = os.path.join(self.image_path, "tile_positions.csv")
        anchor_offset_file = os.path.join(
            self.image_path, "anchor_offsets.csv")
        if os.path.exists(tile_pos_file) and os.path.exists(anchor_offset_file):
            absolute_positions = {}
            anchor_offsets = {}
            with open(tile_pos_file, "r") as f:
                for line in f.readlines():
                    tile = line[0:line.find(':')]
                    pos_x = line[line.find('(') + 1:line.find(',')]
                    pos_y = line[line.find(',') + 1:line.find(')')]
                    absolute_positions[tile] = (int(pos_x), int(pos_y))
            IS_TILE_ZERO_INDEXED = True if str(
                list(absolute_positions.keys())[0]).find("0") else False
            total_tiles_queued = len(absolute_positions.keys())
            with open(anchor_offset_file, "r") as f:
                for line in f.readlines():
                    tile = line[0:line.find(':')]
                    is_col = int(line[line.find(':')+1:line.find('->')])
                    pos_x = line[line.find('(') + 1:line.find(',')]
                    pos_y = line[line.find(',') + 1:line.find(')')]
                    anchor = (int(pos_x), int(pos_y))
                    if not tile in anchor_offsets:
                        anchor_offsets[tile] = {}
                    anchor_offsets[tile][is_col] = anchor

            num_tiles_per_row = NUM_ROWS
            grid_rows = num_tiles_per_row
            grid_cols = int(total_tiles_queued / num_tiles_per_row)
            assert grid_cols % 1 == 0, "Number of tiles not divisible by 'rows'. Check and try again."
            final_image = None
            for y in range(grid_cols):
                tiles_this_row = []
                for x in range(y*grid_rows, (y+1)*grid_rows):
                    if IS_TILE_ZERO_INDEXED:
                        tile = f"tile_{x}.jpg"
                    else:
                        tile = f"tile_{x+1}.jpg"
                    tiles_this_row.append(tile)
                # self.logger.debug(tiles_this_row)
                last_anchor = (0, 0)
                last_pos = (0, 0)
                for i, tile in enumerate(tiles_this_row):
                    this_tile = os.path.join(self.image_path, tile)
                    if i == 0:
                        overlay_image = cv.resize(
                            cv.imread(this_tile), (0, 0), fx=SCALE_BY, fy=SCALE_BY)
                        tile_shape = tuple(overlay_image.shape[0:2][::-1])
                        self.df_prepare(tile_shape)
                        overlay_image = self.df_correct(overlay_image)
                        anchor = anchor_offsets[tile][0]
                        this_position = absolute_positions[tile]
                    else:
                        overlay_image = np.concatenate((overlay_image,
                                                        self.df_correct(
                                                            cv.resize(cv.imread(this_tile),
                                                                      (0, 0),
                                                                      fx=SCALE_BY, fy=SCALE_BY))),
                                                       axis=0)

                        # this_position = absolute_positions[tile]
                        # offset_x = last_pos[0]
                        # offset_y = last_pos[1] + tile_shape[1]
                        # overlay_image = self.overlay_images(
                        #     overlay_image,
                        #     self.df_correct(cv.resize(cv.imread(this_tile), (0, 0),
                        #                               fx=SCALE_BY, fy=SCALE_BY)),
                        #     offset_x,
                        #     offset_y,
                        #     is_tile=True
                        # )
                        # last_pos = (offset_x, offset_y)

                        this_position = absolute_positions[tile]
                        offset_x = last_pos[0]
                        offset_y = last_pos[1]

                        # if first row this is a bit of a waste as both points are the same
                        if tile in anchor_offsets and 0 in anchor_offsets[tile].keys():
                            overlay_shape = tuple(
                                overlay_image.shape[0:2][::-1])
                            anchor = anchor_offsets[tile][0]
                            anchor_pos = (
                                last_anchor[0], overlay_shape[1] - 2 * tile_shape[1] + last_anchor[1])
                            center_x = anchor_pos[0]
                            center_y = anchor_pos[1]
                            offset_x = (this_position[0] - last_pos[0])
                            offset_y = (this_position[1] - last_pos[1])
                            offset = (offset_x, offset_y)
                            target_pos_x = center_x - offset[0]
                            target_pos_y = center_y + tile_shape[1] - offset[1]
                            target_pos = (target_pos_x, target_pos_y)

                            if last_anchor != (-1, -1):
                                # Draw the 'X'
                                cv.line(overlay_image, anchor_pos,
                                        target_pos, (0, 0, 255), 20)
                            # cv.line(overlay_image, (center_x - 200, center_y + 200),
                            #         (center_x + 200, center_y - 200), (0, 0, 255), 20)

                    last_pos = this_position
                    last_anchor = anchor

                    # self.plot_image(overlay_image, (20, 20))
                self.logger.info(f"Stitching column {y+1}")
                if y == 0:
                    final_image = overlay_image
                else:
                    final_image = np.concatenate(
                        (final_image, overlay_image), axis=1)
                    # offset_x = y * tile_shape[0]
                    # offset_y = 0
                    # final_image = self.overlay_images(
                    #     final_image,
                    #     overlay_image,
                    #     offset_x,
                    #     offset_y,
                    #     is_column=True
                    # )
                # last_col_pos = absolute_positions[tiles_this_row[0]]
                # self.plot_image(final_image, (20, 20))
                # out_path = os.path.join(self.image_path, f"column_{y+1}.jpg")
                # cv.imwrite(out_path, overlay_image)

            # Rotate image by +/-1 deg at a time until pink edges are gone
            # final_image = self.rotate_image(final_image)

            # Downscale output image to 5% so it's not as ridiculously huge
            final_image = cv.resize(final_image, (0, 0), fx=0.05, fy=0.05)

            # Save and show output image
            out_path = os.path.join(self.image_path, f"stitched_matrix.jpg")
            self.logger.info(
                f"Saving stitched matrix: {os.path.basename(out_path)}")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            cv.imwrite(out_path, final_image)
            self.plot_image(final_image)
            # cv.imshow("Stitched Matrix", cv.resize(
            #     final_image, (0, 0), fx=0.03, fy=0.03))
            # cv.waitKey(0)
            # cv.destroyAllWindows()
        else:
            raise FileNotFoundError(
                "Tile position and anchor offet files are required.")

    def run(self):  # , image_path = os.getcwd()):

        # Instantiate multiprocessing logger
        self.__log__()

        self.logger.info("Starting...")
        # self.logger.warning('This is a warning')
        # self.logger.error('This is an error')

        offset_x_raw = [-1]
        offset_y_raw = [-1]
        offset_x_avg = 0
        offset_y_avg = 0
        offset_x_std = 5
        offset_y_std = 10

        # allowable_deviation_factor = 4

        overlay_image = None
        is_first_row = True
        offsets_queued = {}
        total_tiles_queued = 0
        sentinel_flags_rxd = 0

        # stores filenames and offsets for column images, queued as processed
        col_files = []
        col_offsets = {}

        tile_shape = (0, 0)

        a_tiles = {}
        tile_relations = {}
        tiles_with_offsets = []
        tiles_with_warnings = []

        tiles, is_good, raw_offsets, used_offsets = [], [], [], []
        anchor_offsets = {}

        num_tiles_per_row = 0

        while True:
            item = self.queue_rx1A.get()
            if item is None:  # Sentinel value to signal end of processing
                break

            self.logger.debug(f"Dequeueing: {item}")

            idx = 0  # number of tiles stitched this row
            # last_tile = ""
            while len(item) > idx:

                # nothing_new = True if self.queue_rx2.empty() else False
                # not_1st_row = True if total_tiles_queued > len(item) else False
                # read_queue2 = True if nothing_new and not_1st_row else False

                if (len(offsets_queued) > idx and
                    total_tiles_queued > len(item) and
                        self.queue_rx2.empty()):
                    self.logger.debug(
                        "Skipping detector queue read to focus on stitching known offsets")
                elif not sentinel_flags_rxd == 2:
                    resp = self.queue_rx2.get()
                    if resp is None:  # Sentinel value to signal end of processing
                        sentinel_flags_rxd += 1
                        self.logger.debug(
                            "All {} offsets detected and queued!".format(
                                "tile" if not sentinel_flags_rxd == 2 else "column"
                            )
                        )
                        continue

                    # Queue tiles and columns offsets separately
                    # For cols, use dictionary with filename as key
                    # and only add entry if not already in the dict
                    (a_tile, tile), offset, anchor = resp
                    t1_num = int(a_tile[a_tile.index("_")+1:-4])
                    t2_num = int(tile[tile.index("_")+1:-4])
                    if abs(t2_num - t1_num) <= 1:
                        if tile.startswith("tile"):
                            offsets_queued[tile] = offset
                            total_tiles_queued += 1
                        elif tile not in col_offsets.keys():
                            # cols are post-processed after all tiles
                            # only take 1st value to avoid all (0, 0) values
                            col_offsets[tile] = offset
                        a_tiles[tile] = a_tile
                    else:
                        valid_offset = True
                        if offset != (0, 0):  # TODO: remove reliance on (0, 0) offsets in queue
                            # if (abs(offset_x_avg - offset[0]) >
                            #     allowable_deviation_factor * offset_x_std):
                            #         valid_offset = False
                            # if (abs(offset_y_avg - offset[1]) >
                            #     allowable_deviation_factor * offset_y_std):
                            #     valid_offset = False
                            check = CheckStd(
                                offset,
                                (offset_y_avg, -offset_x_avg),
                                (offset_y_std, offset_x_std)
                            )
                            valid_offset = all(check)
                        if valid_offset:
                            if not a_tile in tile_relations:
                                tile_relations[a_tile] = {}
                            tile_relations[a_tile][tile] = offset
                            if not tile in tile_relations:
                                tile_relations[tile] = {}
                            tile_relations[tile][a_tile] = tuple(
                                [-x for x in offset])

                    if not a_tile in tiles_with_offsets:
                        tiles_with_offsets.append(a_tile)
                    if not tile in tiles_with_offsets:
                        tiles_with_offsets.append(tile)

                    if not a_tile in anchor_offsets:
                        anchor_offsets[a_tile] = []
                    if anchor is not None:
                        anchor_offsets[a_tile].append(anchor)

                if is_first_row:
                    # Re-evaluate 'is_first_row' only if it is still True
                    # is_first_row = True if total_tiles_queued <= len(item) else False
                    idx = total_tiles_queued - 1  # hold first row processing
                    self.logger.debug(
                        f"Processing first row, setting 'idx = {idx}'")

                end_of_row = True if idx == len(item) - 1 else False

                if is_first_row and not end_of_row:
                    self.logger.debug(
                        "Postponing first row until fully queued")
                    continue  # until entire row of offsets can be compared

                if is_first_row and end_of_row:
                    self.logger.debug("Processing first row now!")
                    is_first_row = False  # once flag is set to False, it will stay False
                    idx = 0  # reset to now process the entire 1st row
                    offset_x_raw, offset_y_raw = [], []
                    for tile, offset in offsets_queued.items():
                        offset_x_raw.append(offset[0])
                        offset_y_raw.append(offset[1])
                    offset_x_raw.sort()
                    offset_y_raw.sort()
                    # self.logger.debug(offset_x_raw)
                    # self.logger.debug(offset_y_raw)
                    offset_x_avg = round(
                        offset_x_raw[int(len(offset_x_raw) / 2)])
                    offset_y_avg = round(
                        offset_y_raw[int(len(offset_y_raw) / 2)])
                    # offset_x_std = round(np.std(offset_x_raw))
                    # offset_y_std = round(np.std(offset_y_raw))
                    offset_x_raw, offset_y_raw = [], []
                    self.logger.debug(
                        f"1st row midpoint offset: ({offset_x_avg}, {offset_y_avg})")
                    self.logger.debug(
                        f"1st row raw offset dev:  ({offset_x_std}, {offset_y_std})")
                    num_tiles_per_row = len(item)
                    self.logger.debug(
                        f"Number of tiles per row: {num_tiles_per_row}")

                if end_of_row:
                    for tile, offset in offsets_queued.items():
                        # if (abs(offset_x_avg - offset[0]) <
                        #     allowable_deviation_factor * offset_x_std):
                        #     offset_x_raw.append(offset[0])
                        # if (abs(offset_y_avg - offset[1]) <
                        #     allowable_deviation_factor * offset_y_std):
                        #     offset_y_raw.append(offset[1])
                        check = CheckStd(
                            offset,
                            (offset_x_avg, offset_y_avg),
                            (offset_x_std, offset_y_std)
                        )
                        if check[0]:
                            offset_x_raw.append(offset[0])
                        if check[1]:
                            offset_y_raw.append(offset[1])
                    offset_x_avg = round(np.average(offset_x_raw))
                    offset_y_avg = round(np.average(offset_y_raw))
                    offset_x_std = round(np.std(offset_x_raw))
                    offset_y_std = round(np.std(offset_y_raw))
                    self.logger.debug(
                        f"Row-end average offset: ({offset_x_avg}, {offset_y_avg})")
                    self.logger.debug(
                        f"Row-end offset stddev:  ({offset_x_std}, {offset_y_std})")
                    resp = [(offset_x_avg, offset_y_avg),
                            (offset_x_std, offset_y_std)]
                    self.queue_tx3.put(resp)

                if item[idx] in offsets_queued.keys():
                    # Stitch tiles together with given offset
                    tile = item[idx]
                    a_tile = a_tiles[tile]
                    offset = offsets_queued[tile]
                    this_tile = os.path.join(self.image_path, tile)

                    valid_offset = True
                    if offset != (0, 0):  # TODO: remove reliance on (0, 0) offsets in queue
                        # if (abs(offset_x_avg - offset[0]) >
                        #     allowable_deviation_factor * offset_x_std):
                        #         valid_offset = False
                        # if (abs(offset_y_avg - offset[1]) >
                        #     allowable_deviation_factor * offset_y_std):
                        #     valid_offset = False
                        check = CheckStd(
                            offset,
                            (offset_x_avg, offset_y_avg),
                            (offset_x_std, offset_y_std)
                        )
                        valid_offset = all(check)
                    if valid_offset:
                        if not a_tile in tile_relations:
                            tile_relations[a_tile] = {}
                        tile_relations[a_tile][tile] = offset
                        if not tile in tile_relations:
                            tile_relations[tile] = {}
                        tile_relations[tile][a_tile] = tuple(
                            [-x for x in offset])
                        self.logger.debug(  # TODO: Make 'info'
                            f"Using abs offset {offset} for {tile}")
                    else:
                        self.logger.warning(
                            f"Invalid offset {offset} for {tile}")
                        offset = (offset_x_avg, offset_y_avg)
                        self.logger.debug(  # TODO: Make 'info'
                            f"Using avg offset {offset} for {tile}")

                    # DEBUG: Track good, raw, used offsets
                    tiles.append(tile)
                    is_good.append(valid_offset)
                    raw_offsets.append(offsets_queued[tile])
                    used_offsets.append(offset)

                    # x, y = offset
                    # if offset == (0, 0):
                    #     # first tile in row can be processed immediately
                    #     idx += 1
                    #     last_tile = tile
                    #     continue
                    # if total_tiles_queued < 3:
                    #     # wait until 3 tiles are queued to confirm consistency
                    #     continue
                    # elif offset == (-1, -1):
                    #     # take the average when no key points found
                    #     x = offset_x_avg
                    #     y = offset_y_avg
                    # elif offset_x_raw == [-1] or offset_y_raw == [-1]:
                    #     if offset_x_raw == [-1]:
                    #         offset_x_raw = [x]
                    #     if offset_y_raw == [-1]:
                    #         offset_y_raw = [y]
                    # else:
                    #     offset_x_raw.append(x)
                    #     offset_y_raw.append(y)
                    #     offset_x_avg = np.average(offset_x_raw)
                    #     offset_y_avg = np.average(offset_y_raw)
                    #     offset_x_std = np.std(offset_x_raw)
                    #     offset_y_std = np.std(offset_y_raw)
                    # time.sleep((idx + 1) * 0.05)

                    # Append tile_image to overlay_image
                    self.logger.info(f"Processing {tile}")
                    if offset == (0, 0):  # TODO: remove reliance on (0, 0) offsets in queue
                        overlay_image = cv.resize(
                            cv.imread(this_tile), (0, 0), fx=SCALE_BY, fy=SCALE_BY)
                        tile_shape = tuple(overlay_image.shape[0:2][::-1])
                    else:
                        overlay_image = self.overlay_images(
                            overlay_image,
                            cv.resize(cv.imread(this_tile), (0, 0),
                                      fx=SCALE_BY, fy=SCALE_BY),
                            offset[0],
                            offset[1]
                        )
                    idx += 1
                    # last_tile = tile

            # # NOTE: Save overlay_image to file (column), store global column file list,
            # # and add overlapping column images to detector queue as they are generated
            # col_filename = f"column_{len(col_files)+1}.png"
            # self.logger.info(f"Saving image: {col_filename}")
            # full_col_file = os.path.join(self.image_path, col_filename)
            # cv.imwrite(full_col_file, overlay_image)
            # col_files.append(col_filename)
            # # col_offsets[col_filename] = (-offset_y_avg, offset_x_avg)
            # if len(col_files) >= 2:
            #     # Ask detector to process offsets for last vs new column image
            #     # and queue the results in 'col_offsets' for post-processing
            #     self.queue_rx1B.put(col_files[-2:])
            #     pass

            # remove keys in 'item' that exist in 'offsets_queued'
            self.logger.debug(f"len(offsets_queued) = {len(offsets_queued)}")
            self.logger.debug("Flushing processed offsets from queue...")
            for i in item:
                if i in offsets_queued.keys():
                    offsets_queued.pop(i)
            self.logger.debug(f"len(offsets_queued) = {len(offsets_queued)}")

        # NOTE: No TX queues to wait on in this process

        # Indicate to detector that we are done passing columns to it
        # Detector will die after 2nd sentinel flag is received, not just one
        self.queue_rx1B.put(None)

        while True:
            if sentinel_flags_rxd == 2:
                break
            resp = self.queue_rx2.get()
            if resp is None:  # Sentinel value to signal end of processing
                sentinel_flags_rxd += 1
                self.logger.debug(
                    "All {} offsets detected and queued!".format(
                        "tile" if not sentinel_flags_rxd == 2 else "column"
                    )
                )
                continue

            # Queue tiles and columns offsets separately
            # For cols, use dictionary with filename as key
            # and only add entry if not already in the dict
            (a_tile, tile), offset, anchor = resp
            t1_num = int(a_tile[a_tile.index("_")+1:-4])
            t2_num = int(tile[tile.index("_")+1:-4])
            if abs(t2_num - t1_num) <= 1:
                if tile.startswith("tile"):
                    offsets_queued[tile] = offset
                    total_tiles_queued += 1
                elif tile not in col_offsets.keys():
                    # cols are post-processed after all tiles
                    # only take 1st value to avoid all (0, 0) values
                    col_offsets[tile] = offset
                a_tiles[tile] = a_tile
            else:
                valid_offset = True
                if offset != (0, 0):  # TODO: remove reliance on (0, 0) offsets in queue
                    # if (abs(offset_x_avg - offset[0]) >
                    #     allowable_deviation_factor * offset_x_std):
                    #         valid_offset = False
                    # if (abs(offset_y_avg - offset[1]) >
                    #     allowable_deviation_factor * offset_y_std):
                    #     valid_offset = False
                    check = CheckStd(
                        offset,
                        (offset_y_avg, -offset_x_avg),
                        (offset_y_std, offset_x_std)
                    )
                    valid_offset = all(check)
                if valid_offset:
                    if not a_tile in tile_relations:
                        tile_relations[a_tile] = {}
                    tile_relations[a_tile][tile] = offset
                    if not tile in tile_relations:
                        tile_relations[tile] = {}
                    tile_relations[tile][a_tile] = tuple([-x for x in offset])

            if not a_tile in tiles_with_offsets:
                tiles_with_offsets.append(a_tile)
            if not tile in tiles_with_offsets:
                tiles_with_offsets.append(tile)

            if not a_tile in anchor_offsets:
                anchor_offsets[a_tile] = []
            anchor_offsets[a_tile].append(anchor)

        # DEBUG: Print statistics on good, raw, used offsets
        self.logger.debug("Offset statistics:")
        out = np.column_stack((tiles, is_good, raw_offsets, used_offsets))
        for line in out:
            self.logger.debug(line)
        count_good = is_good.count(True)
        count_bad = is_good.count(False)
        self.logger.debug(
            f"Tiles with 'bad' offsets: {count_bad} / {count_good+count_bad}")

        self.logger.debug(col_offsets)

        restore_x_offset_avg = offset_x_avg
        restore_y_offset_avg = offset_y_avg

        # # TODO: Stitch column images together into overlay image, rotate, and crop
        # # Do this by processing the generated global column files and offset lists
        # try:
        #     offset_x_raw, offset_y_raw = [], []
        #     for offset in col_offsets.values():
        #         offset_x_raw.append(offset[0])
        #         offset_y_raw.append(offset[1])
        #     offset_x_raw.sort()
        #     offset_y_raw.sort()
        #     # self.logger.debug(offset_x_raw)
        #     # self.logger.debug(offset_y_raw)
        #     offset_x_avg = round(offset_x_raw[int(len(offset_x_raw) / 2)])
        #     offset_y_avg = round(offset_y_raw[int(len(offset_y_raw) / 2)])
        #     # offset_x_std = round(np.std(offset_x_raw))
        #     # offset_y_std = round(np.std(offset_y_raw))
        #     offset_m_std = 2*max(offset_x_std, offset_y_std)
        #     self.logger.info(
        #         f"Column average = {(offset_x_avg, offset_y_avg)}, STD = {offset_m_std}")
        #     offset_x_raw, offset_y_raw = [], []
        #     col_offsets_to_correct = []
        #     for i, offset in enumerate(col_offsets.values()):
        #         if i == 0 and offset == (0, 0):
        #             continue
        #         check = CheckStd(
        #             offset,
        #             (offset_x_avg, offset_y_avg),
        #             (offset_m_std, offset_m_std)
        #             # NOTE: Use X/Y max STD values for col stitching
        #         )
        #         if check[0]:
        #             offset_x_raw.append(offset[0])
        #         if check[1]:
        #             offset_y_raw.append(offset[1])
        #         if not all(check):
        #             col_offsets_to_correct.append(offset)
        #     offset_x_avg = round(np.average(offset_x_raw))
        #     offset_y_avg = round(np.average(offset_y_raw))
        # except:
        #     self.logger.error(
        #         "Unable to correct column offset outliers using averages",
        #         exc_info=1
        #     )
        #     col_offsets_to_correct = col_offsets.values()[1:]
        #     offset_x_avg = restore_y_offset_avg  # invert XY
        #     offset_y_avg = restore_x_offset_avg  # invert XY

        # overlay_image = None
        # for i, col_filename in enumerate(col_offsets.keys()):
        #     self.logger.info(f"Stitching column {i+1}")
        #     this_column = os.path.join(self.image_path, col_filename)
        #     offset = col_offsets[col_filename]
        #     if offset in col_offsets_to_correct:
        #         offset = (offset_x_avg, offset_y_avg)
        #         self.logger.warning(
        #             f"Column {i+1} offset corrected from {col_offsets[col_filename]} to {offset}"
        #         )
        #     if i == 0:
        #         overlay_image = cv.imread(this_column)
        #     else:
        #         overlay_image = self.overlay_images(
        #             overlay_image,
        #             cv.imread(this_column),
        #             offset[0],
        #             offset[1]
        #         )

        # # Check for large image and resize if needed
        # image_size = len(overlay_image.flatten())
        # max_size = CV_IO_MAX_IMAGE_PIXELS
        # if image_size > max_size:
        #     # cv.resize(image, (0,0), fx=0.5, fy=0.5)
        #     # overlay_image = cv.resize(overlay_image, (2^15, 2^15))
        #     scale_by = int(100 * max_size / image_size) / 100
        #     self.logger.info(f"Resizing output, scale = {scale_by}x")
        #     overlay_image = cv.resize(
        #         overlay_image, (0, 0), fx=scale_by, fy=scale_by)

        # # Rotate image by +/-1 deg at a time until pink edges are gone
        # overlay_image = self.rotate_image(overlay_image)

        # # Save final output image
        # full_out_path = os.path.join(self.image_path, f"stitched_out.jpg")
        # self.logger.info(f"Saving image: {os.path.basename(full_out_path)}")
        # cv.imwrite(full_out_path, overlay_image)

        IS_TILE_ZERO_INDEXED = True if str(tiles[0]).find("0") else False
        self.logger.info("Tile offsets, in order:")
        absolute_positions = {}
        fixed_tiles = []
        loop_count = 0
        its_the_final_countdown = False
        double_reverse = False
        while True:
            loop_count += 1
            tiles_found_this_round = 0
            for i in range(total_tiles_queued):
                if IS_TILE_ZERO_INDEXED:
                    tile = f"tile_{i}.jpg"
                else:
                    tile = f"tile_{i+1}.jpg"
                if not tile in tile_relations:
                    if loop_count == 1:
                        self.logger.info(f"{tile} -> None")
                    # tile_relations[tile] = (-1, -1) # TODO use average instead, or omit it entirely
                else:
                    if loop_count == 1:
                        self.logger.info(f"{tile} -> {tile_relations[tile]}")
                    for a_tile, offset in tile_relations[tile].items():
                        a_tile = str(a_tile)
                        if offset == (0, 0) and len(fixed_tiles) == 0:
                            absolute_positions[tile] = offset
                            fixed_tiles.append(tile)
                        if tile in fixed_tiles:
                            if not a_tile in fixed_tiles:
                                abs_x = absolute_positions[tile][0] + offset[0]
                                abs_y = absolute_positions[tile][1] + offset[1]
                                absolute_positions[a_tile] = (abs_x, abs_y)
                                fixed_tiles.append(a_tile)
                                tiles_found_this_round += 1
                                if its_the_final_countdown:
                                    self.logger.warning(
                                        f"Roughly placed {tile} after using an average offset")
                            else:
                                now_x = absolute_positions[a_tile][0]
                                now_y = absolute_positions[a_tile][1]
                                abs_x = absolute_positions[tile][0] + offset[0]
                                abs_y = absolute_positions[tile][1] + offset[1]
                                if abs(now_x - abs_x) > 10 or abs(now_y - abs_y) > 10:
                                    if a_tile not in tiles_with_warnings:
                                        tiles_with_warnings.append(a_tile)
                                    self.logger.warning(f"Large position difference for {a_tile}: "
                                                        f"({now_x}, {now_y}) -> ({abs_x}, {abs_y})")
                                abs_x = int((now_x + abs_x) / 2)
                                abs_y = int((now_y + abs_y) / 2)
                                absolute_positions[a_tile] = (abs_x, abs_y)
                        else:
                            # self.logger.debug(
                            #     f"Deferring unanchored tile: {tile}")
                            break
            self.logger.debug(
                f"Found {tiles_found_this_round} tiles on loop #{loop_count}")
            if tiles_found_this_round == 0:
                its_the_final_countdown = True
                reverse_search = False
                for i in range(total_tiles_queued):
                    if IS_TILE_ZERO_INDEXED:
                        last_tile = f"tile_{i-1}.jpg"
                        tile = f"tile_{i}.jpg"
                    else:
                        last_tile = f"tile_{i}.jpg"
                        tile = f"tile_{i+1}.jpg"
                    num_tiles_per_row = NUM_ROWS
                    grid_rows = num_tiles_per_row
                    image_row = i % grid_rows
                    if reverse_search == False and not tile in absolute_positions.keys():
                        # NOTE: This position estimation will fail if it's the first tile in a row
                        if image_row > 0 and last_tile in absolute_positions.keys():
                            previous_position = absolute_positions[last_tile]
                            abs_x = previous_position[0] + restore_x_offset_avg
                            abs_y = previous_position[1] + restore_y_offset_avg
                            absolute_positions[tile] = (abs_x, abs_y)
                            self.logger.warning(
                                f"Used an average position offset to place {tile}")
                            break  # only set one tile with average position per no new tiles being placed
                        else:
                            if double_reverse and image_row == 0:
                                if IS_TILE_ZERO_INDEXED:
                                    last_tile = f"tile_{i-grid_rows}.jpg"
                                else:
                                    last_tile = f"tile_{i+1-grid_rows}.jpg"
                                previous_position = absolute_positions[last_tile]
                                # invert x->y
                                abs_x = previous_position[0] + \
                                    restore_y_offset_avg
                                # invert row->col
                                abs_y = previous_position[1] - \
                                    restore_x_offset_avg
                                absolute_positions[tile] = (abs_x, abs_y)
                                self.logger.warning(
                                    f"Used an row -> col position averaging for {tile}")
                                double_reverse = False
                                break
                            reverse_search = True
                            self.logger.warning(
                                f"Reversing search order: {reverse_search}")
                    if reverse_search == True and not last_tile in absolute_positions.keys():
                        # NOTE: This position estimation will fail if it's the first tile in a row
                        if image_row > 0 and tile in absolute_positions.keys():
                            previous_position = absolute_positions[tile]
                            abs_x = previous_position[0] - restore_x_offset_avg
                            abs_y = previous_position[1] - restore_y_offset_avg
                            absolute_positions[last_tile] = (abs_x, abs_y)
                            self.logger.warning(
                                f"Used an average position offset to place {last_tile}")
                            break  # only set one tile with average position per no new tiles being placed
                        else:
                            double_reverse = True
                            self.logger.warning(
                                f"Reversing search order: {reverse_search}")
                            break
            if len(absolute_positions) == total_tiles_queued:
                break
            if loop_count > total_tiles_queued:
                self.logger.error(
                    "Too many loop iterations trying to position tiles!")
                break

        self.logger.info("Absolute positions:")
        for tile, pos in absolute_positions.items():
            self.logger.info(f"{tile} -> {pos}")

        # Find tiles with no position found
        self.logger.info("Tiles with no position found:")
        tiles_found = False
        for i in range(total_tiles_queued):
            if IS_TILE_ZERO_INDEXED:
                last_tile = f"tile_{i-1}.jpg"
                tile = f"tile_{i}.jpg"
            else:
                last_tile = f"tile_{i}.jpg"
                tile = f"tile_{i+1}.jpg"
            if not tile in absolute_positions.keys():
                # TODO: This position estimation will fail if it's the first tile in a row
                previous_position = absolute_positions[last_tile]
                abs_x = previous_position[0] + restore_x_offset_avg
                abs_y = previous_position[1] + restore_y_offset_avg
                absolute_positions[tile] = (abs_x, abs_y)
                self.logger.error(f"> {tile}")
                tiles_found = True
        if not tiles_found:
            self.logger.info("> NONE")

        # Find tiles with no offsets found
        self.logger.info("Tiles with no absolute offsets found:")
        tiles_found = False
        for i in range(total_tiles_queued):
            if IS_TILE_ZERO_INDEXED:
                tile = f"tile_{i}.jpg"
            else:
                tile = f"tile_{i+1}.jpg"
            if not tile in tile_relations:
                self.logger.error(f"> {tile}")
                tiles_found = True
        if not tiles_found:
            self.logger.info("> NONE")

        # Find tiles with offset warnings
        self.logger.info("Tiles with large offset errors:")
        tiles_found = False
        for i in range(total_tiles_queued):
            if IS_TILE_ZERO_INDEXED:
                tile = f"tile_{i}.jpg"
            else:
                tile = f"tile_{i+1}.jpg"
            if tile in tiles_with_warnings:
                self.logger.error(f"> {tile}")
                tiles_found = True
        if not tiles_found:
            self.logger.info("> NONE")

        tile_pos_file = os.path.join(self.image_path, "tile_positions.csv")
        with open(tile_pos_file, "w") as f:
            for i in range(total_tiles_queued):
                if IS_TILE_ZERO_INDEXED:
                    tile = f"tile_{i}.jpg"
                else:
                    tile = f"tile_{i+1}.jpg"
                f.write(f"{tile}:{absolute_positions[tile]}\n")

        anchor_offset_file = os.path.join(
            self.image_path, "anchor_offsets.csv")
        with open(anchor_offset_file, "w") as f:
            # sort dictionary alphabetically by keys first
            # anchor_offsets = dict(sorted(anchor_offsets.items()))
            # write out the sorted anchor offsets to file:
            for tile, anchors in anchor_offsets.items():
                try:
                    tile_num = int(tile[5:-4])
                    if IS_TILE_ZERO_INDEXED:
                        image_row = ((tile_num + 1) % NUM_ROWS)
                    else:
                        image_row = (tile_num % NUM_ROWS)
                    # image_col = int(tile_num / NUM_ROWS)
                    start_idx = 1 if image_row == 0 else 0
                    for x, y in enumerate(anchors):
                        for is_col, anchor in y.items():
                            f.write(f"{tile}:{start_idx+x}->{anchor}\n")
                except Exception as e:
                    f.write(f"ERROR: {tile} -> {anchors}\n")
                    self.logger.error(
                        "Error writing anchor offsets", exc_info=True)

        # grid_rows = num_tiles_per_row
        # grid_cols = total_tiles_queued / num_tiles_per_row
        # assert grid_cols % 1 == 0, "Number of tiles not divisible by 'rows'. Check and try again."
        # for i in range(total_tiles_queued):
        #     tile = f"tile_{i+1}.jpg"
        #     this_tile = os.path.join(self.image_path, tile)
        #     if i == 0:
        #         overlay_image = cv.imread(this_tile)
        #         tile_shape = tuple(overlay_image.shape[0:2][::-1])
        #     else:
        #         overlay_shape = tuple(overlay_image.shape[0:2][::-1])
        #         this_position = absolute_positions[tile]
        #         offset_x = -(overlay_shape[0] - tile_shape[0]) + this_position[0]
        #         offset_y = -(overlay_shape[1] - tile_shape[1]) + this_position[1]
        #         overlay_image = self.overlay_images(
        #             overlay_image,
        #             cv.imread(this_tile),
        #             offset_x,
        #             offset_y
        #         )
        #     self.plot_image(overlay_image, (20, 20))
        # out_path = os.path.join(self.image_path, "stitched_absolute.jpg")
        # cv.imwrite(out_path, overlay_image)

        # Use absolute positioning to place tiles
        self.place_tiles_absolutely()

        self.logger.info("Finished!")
        return

        # Load the images
        image_paths = [path for path in os.listdir(image_path) if path.endswith(
            "jpg") and not path.startswith("stitched")]
        image_ids = [int(path[path.rindex("_")+1:-4]) for path in image_paths]
        sort_order = np.argsort(image_ids)
        sorted_paths = np.array(image_paths)[sort_order]
        image_ids = np.array(image_ids)[sort_order]

        # TODO calculate 'grid_rows' automatically, somehow
        total_tiles = len(image_ids)
        grid_rows = 22  # vertical tile count, Y-axis
        grid_cols = total_tiles / grid_rows  # horizontal tile count, X-axis
        assert grid_cols % 1 == 0, "Number of tiles not divisible by 'rows'. Check and try again."
        grid_cols = int(grid_cols)

        image_row = [(id - 1) % grid_rows for id in image_ids]
        image_col = [int((id - 1) / grid_rows) for id in image_ids]

        image_pos = []
        for i in range(len(image_ids)):
            image_pos.append({"col": image_col[i], "row": image_row[i]})

        # Define the VERTICAL stitching offsets
        row_point1_x = 735  # X pixel of rows calibration point in reference image 1
        row_point1_y = 681  # Y pixel of rows calibration point in reference image 1
        row_point2_x = 783  # X pixel of rows calibration point in reference image 2
        row_point2_y = 12   # Y pixel of rows calibration point in reference image 2

        # Define the HORIZONTAL stitching offsets
        col_point1_x = 1771  # X pixel of columns calibration point in reference image 1
        col_point1_y = 1674  # Y pixel of columns calibration point in reference image 1
        col_point2_x = 1100  # X pixel of columns calibration point in reference image 2
        col_point2_y = 1627  # Y pixel of columns calibration point in reference image 2

        # Define tile sizes at runtime
        tile_width = 0
        tile_height = 0
        column_width = 0
        column_height = 0

        full_sorted_paths = [os.path.join(
            image_path, file) for file in sorted_paths]
        main_image = None
        main_position = {"col": -1, "row": -1}
        stitched_col_files = []  # TODO comment out next 2 lines!!!

        DEBUG = False
        SKIP_COL_CREATE = False

        if SKIP_COL_CREATE:
            stitched_col_files = [path for path in os.listdir(
                image_path) if path.startswith("stitch") and not path.endswith("out.jpg")]
            stitched_col_files = [os.path.join(
                image_path, path) for path in stitched_col_files]

        for i in range(len(sorted_paths)):
            if SKIP_COL_CREATE:
                break

            next_image = cv.imread(full_sorted_paths[i])
            next_position = image_pos[i]

            if 0 in (tile_width, tile_height):

                tile_width = next_image.shape[1]
                tile_height = next_image.shape[0]

                row_x_offset = (row_point1_x - row_point2_x) - tile_width
                row_y_offset = (row_point1_y - row_point2_y) - tile_height

            # print(main_position, next_position)
            if main_position["col"] == next_position["col"]:
                # Overlay the images
                main_image = self.overlay_images(main_image,
                                                 next_image,
                                                 # - next_image.shape[1],
                                                 row_x_offset,
                                                 row_y_offset)  # - next_image.shape[0])
            if main_position["col"] != next_position["col"] or i == len(sorted_paths) - 1:
                if i == len(sorted_paths) - 1:
                    print("Processed", sorted_paths[i])
                if main_position["col"] >= 0:
                    if DEBUG:
                        self.plot_image(main_image, (5, 20))

                    # Use a lossless image file format (i.e. png) to avoid compression artifacts along edges
                    full_out_path = os.path.join(
                        image_path, f"stitched_col{main_position['col']+1:02.0f}.png")
                    print("Creating", os.path.basename(full_out_path))
                    main_image = cv.cvtColor(main_image, cv.COLOR_BGR2RGB)
                    cv.imwrite(full_out_path, main_image)
                    stitched_col_files.append(full_out_path)

                main_image = next_image
                main_position = image_pos[i]
            else:
                print("Processed", sorted_paths[i])

        first_iter = True
        main_image = None
        num_col_files = len(stitched_col_files)

        DEBUG = False

        for col_img_path in stitched_col_files:
            print("Processing", os.path.basename(col_img_path))
            next_image = cv.imread(col_img_path)

            if 0 in (column_width, column_height):

                column_width = next_image.shape[1]
                column_height = next_image.shape[0]

                col_x_offset = (col_point1_x - col_point2_x) - \
                    column_width  # width of stitched_col image
                col_y_offset = (col_point1_y - col_point2_y) - \
                    column_height  # height of stitched_col img

            # Create a mask for the specified color
            # mask = np.all(next_image == self.trans_color, axis=-1)

            # Create a 4-channel image (BGRA) with an alpha channel
            # next_image_with_alpha = cv.cvtColor(next_image, cv.COLOR_BGR2BGRA)

            # Set the alpha channel to 0 where the mask is True
            # next_image_with_alpha[mask, 3] = 0

            if DEBUG and False:
                self.plot_image(next_image, (5, 20))

            if first_iter:
                main_image = next_image  # _with_alpha
            else:
                main_image = self.overlay_images(main_image,
                                                 next_image,  # _with_alpha,
                                                 # - next_image.shape[1],
                                                 col_x_offset,
                                                 col_y_offset)  # - next_image.shape[0])
            if DEBUG and not first_iter:
                self.plot_image(main_image, (20, 20))

            first_iter = False

        # Check for large image and resize if needed
        image_size = len(main_image.flatten())
        max_size = CV_IO_MAX_IMAGE_PIXELS
        if image_size > max_size:
            # cv.resize(image, (0,0), fx=0.5, fy=0.5)
            # main_image = cv.resize(main_image, (2^15, 2^15))
            scale_by = int(100 * max_size / image_size) / 100
            print(f"Resizing output, scale = {scale_by}x")
            main_image = cv.resize(
                main_image, (0, 0), fx=scale_by, fy=scale_by)

        # Rotate image by +/-1 deg at a time until pink edges are gone
        main_image = self.rotate_image(main_image)

        # Save final output image
        full_out_path = os.path.join(image_path, f"stitched_alpha_out.jpg")
        print("Creating", os.path.basename(full_out_path))
        cv.imwrite(full_out_path, main_image)
        print("Finished!")

        if DEBUG:
            self.plot_image(main_image, (20, 20))


# if __name__ == "__main__":
#     root = os.path.join(os.getcwd(), "../../content/images")
#     #root = os.path.join(root, "test_matrix")
#     root = os.path.join(root, "raw_images")
#     stitch = FeatureStitcher(None)
#     stitch.run(root)

if __name__ == '__main__':
    proc = ProcessStitcher(None, None, None, None, None)
    path = os.path.dirname(os.getcwd())
    path = os.path.dirname(path)
    path = os.path.join(path, "content\\images\\raw_images\\")
    proc.config(path)
    proc.place_tiles_absolutely()
