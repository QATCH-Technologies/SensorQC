from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, \
    QLabel, QProgressBar, QLineEdit, QPushButton, QFileDialog, \
    QVBoxLayout, QHBoxLayout
from matplotlib import pyplot as plt
from logging.handlers import QueueHandler

import cv2 as cv
import logging
import numpy as np
import os
import sys

DO_BLEND = True
CORRECT_DF = False


class WindowStitcher(QWidget):

    def __init__(self, log_level=logging.DEBUG):
        super().__init__()

        self.queue_log = None
        self.log_level = log_level
        self.__log__()

        self.logger.debug("Stitcher window initializing...")

        self.trans_color = (255, 255, 255)
        self.cached_image = None
        self.image_path = None
        self.running = False
        self.create_window()

        self.logger.debug("Stitcher window initialized")

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

    def setLogLevel(self, log_level):
        """ Set logging level """

        self.log_level = log_level
        self.logger.setLevel(self.log_level)
        self.logger.debug(f"Log level changed to {log_level}")

    def config(self, image_path):
        """ Global configuration parameters common to all children processes """

        self.image_path = image_path

        if self.image_path is not None:
            self.tiles_per_row.setText("0")  # auto-detect
            self.cached_image = None

            if not self.isVisible():
                self.show()

            self.create_image(True)

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

        if self.place_tiles_absolutely_called:
            self.plot_image(out1)
            self.plot_image(out2)
            self.plot_image(out)

        # if BLEND_OVERLAP == BlendType.NONE:
        #     return out

        try:
            # Blend the foreground image onto the background image
            bg_roi = out1[y1+top1:y2, x1+left1:x2]
            fg_roi = fg[max(0, -y_offset):fg.shape[0]-max(0, y_offset),
                        max(0, -x_offset):fg.shape[1]-max(0, x_offset)]

            # if BLEND_OVERLAP == BlendType.ALPHA_50_50:
            #     alpha = 0.5  # Transparency level
            #     blended_roi = cv.addWeighted(bg_roi, alpha, fg_roi, alpha, 0)

            if True:  # elif BLEND_OVERLAP == BlendType.LINEAR_BLEND:
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

    def create_window(self):

        # Master grid layout for main widget
        v_layout = QVBoxLayout()
        h_layout = QHBoxLayout()

        b_load = QPushButton("Load")
        b_load.clicked.connect(self.load_tiles)

        # First row, offset coords, and overlap coords
        self.tiles_per_row = QLineEdit("0")
        self.overlap_x = QLineEdit("0")
        self.overlap_y = QLineEdit("0")
        self.rotate_by = QLineEdit("0")

        b_update = QPushButton("Update")
        b_update.clicked.connect(self.create_image)

        b_save = QPushButton("Save")
        b_save.clicked.connect(self.save_image)

        self.output_size = QLabel("Output Size:")
        self.output_size.setAlignment(Qt.AlignCenter)
        self.output_size.setVisible(False)  # hide

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.reset()

        self.image = QLabel()
        self.image.setContentsMargins(0, 0, 0, 0)
        self.image.setAlignment(Qt.AlignCenter)

        h_layout.addStretch()
        h_layout.addWidget(b_load)
        h_layout.addStretch()
        h_layout.addWidget(QLabel("Tiles Per Row ="))
        h_layout.addWidget(self.tiles_per_row)
        h_layout.addStretch()
        h_layout.addWidget(QLabel("Overlap: X ="))
        h_layout.addWidget(self.overlap_x)
        h_layout.addWidget(QLabel("px  Y ="))
        h_layout.addWidget(self.overlap_y)
        h_layout.addWidget(QLabel("px"))
        h_layout.addStretch()
        h_layout.addWidget(QLabel("Rotate:"))
        h_layout.addWidget(self.rotate_by)
        h_layout.addWidget(QLabel("deg"))
        h_layout.addStretch()
        h_layout.addWidget(b_update)
        h_layout.addWidget(b_save)

        v_layout.addLayout(h_layout)
        v_layout.addStretch()
        v_layout.addWidget(self.output_size)
        v_layout.addWidget(self.progress)
        v_layout.addWidget(self.image)
        v_layout.addStretch()

        # Create a QWidget (the base class for all UI objects in PyQt)
        self.setWindowTitle("Tile Stitcher")
        self.setGeometry(100, 100, 900, 900)  # (x, y, width, height)
        self.setLayout(v_layout)

        self.toolbar = h_layout

    def load_tiles(self):
        folder = str(QFileDialog.getExistingDirectory(
            self, "Select Tiles Path"))
        if len(folder) == 0:
            self.logger.debug("User hit cancel.")
        else:
            self.config(folder)

    def set_image(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height,
                         bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.image.setPixmap(pixmap)
        self.image.resize(pixmap.width(), pixmap.height())

    def save_image(self):
        if self.cached_image is None:
            self.logger.warning(
                "Nothing to save! Click \"Update\" to create an image first.")
        else:
            self.setToolbarEnabled(False)
            self.output_size.setText("Saving image to file...")
            QApplication.processEvents()
            output_path = os.path.join(self.image_path, "stitched_output.jpg")
            cv.imwrite(output_path, self.cached_image)
            self.output_size.setText("Saved!")
            self.setToolbarEnabled(True)

    def create_image(self, quick=False):
        if self.image_path is None:
            self.logger.warning("No path selected. Load a path first!")
            return

        if self.running:
            self.logger.warning("User stopped the image creation!")
            self.output_size.setText("Stopped!")
            self.running = False  # Used as an abort flag
            return

        self.setToolbarEnabled(False, allow_stop=True)
        overlap_x = int(self.overlap_x.text())
        overlap_y = int(self.overlap_y.text())
        rotation = float(self.rotate_by.text())  # deg
        scale_factor = 1.0
        show_factor = 0.05
        if quick:
            scale_factor = show_factor
            overlap_x = int(overlap_x * scale_factor)
            overlap_y = int(overlap_y * scale_factor)
        image_paths = [path for path in os.listdir(self.image_path) if path.endswith(
            "jpg") and not path.startswith("stitched")]
        total_tiles = len(image_paths)

        image_ids = []
        rows = []
        # convert new tile name format to tile number, in processing order
        if len(image_paths) and image_paths[0].index("_") != image_paths[0].rindex("_"):
            for path in image_paths:
                try:
                    row = int(path[path.index("_")+1:path.rindex("_")])
                    col = int(path[path.rindex("_")+1:-4])
                except Exception as e:
                    self.logger.error(
                        "Unable to parse tile position", exc_info=True)
                    row, col = (-1, -1)
                image_ids.append((row, col))
                if not row in rows:
                    rows.append(row)
            NUM_ROWS = len(rows)
            if int(self.tiles_per_row.text()) == 0:
                # do not "auto detect" again later
                self.tiles_per_row.setText(str(NUM_ROWS))
            for i in range(total_tiles):
                row, col = image_ids[i]
                id = row + col * NUM_ROWS
                image_ids[i] = id
        else:
            image_ids = [int(path[path.rindex("_")+1:-4])
                         for path in image_paths]

        sort_order = np.argsort(image_ids)
        sorted_paths = np.array(image_paths)[sort_order]
        image_ids = np.array(image_ids)[sort_order]
        IS_TILE_ZERO_INDEXED = True if str(
            sorted_paths[0]).find("0") else False
        final_image = None
        try:
            if int(self.tiles_per_row.text()) == 0:
                NUM_ROWS = int(np.ceil(np.sqrt(len(sorted_paths))))
                if len(sorted_paths) % NUM_ROWS != 0:
                    NUM_ROWS -= 1  # if not square, assume more cols than rows
                self.tiles_per_row.setText(str(NUM_ROWS))
            else:
                NUM_ROWS = int(self.tiles_per_row.text())
        except:
            self.logger.error(
                f"Cannot parse \"tiles per row\" value: \"{self.tiles_per_row.text()}\"")

        # show progress bar
        self.progress.setRange(0, len(sorted_paths))
        self.output_size.setVisible(False)
        self.progress.setVisible(True)

        crop_pixels = 0
        for i, tile in enumerate(sorted_paths):
            self.progress.setValue(i)  # update progress
            QApplication.processEvents()

            if not self.running:
                self.logger.debug("Stopping task, due to user abort...")
                break

            full_path = os.path.join(self.image_path, tile)
            this_tile_adjusted = cv.resize(cv.imread(full_path), (0, 0),
                                           fx=scale_factor, fy=scale_factor)
            if CORRECT_DF:
                if i == 0:
                    tile_shape = tuple(this_tile_adjusted.shape[0:2][::-1])
                    self.df_prepare(tile_shape)
                this_tile_adjusted = self.df_correct(this_tile_adjusted)

            if rotation != 0:
                (h, w) = this_tile_adjusted.shape[:2]
                (cX, cY) = (w // 2, h // 2)
                M = cv.getRotationMatrix2D((cX, cY), rotation, 1.0)
                this_tile_adjusted = cv.warpAffine(
                    this_tile_adjusted, M, (w, h), borderMode=cv.BORDER_CONSTANT, borderValue=self.trans_color)
                if i == 0:
                    for px in range(min(w, h)):
                        if tuple(this_tile_adjusted[:, 0][px]) != self.trans_color:
                            crop_pixels = px
                            break
                this_tile_adjusted = this_tile_adjusted[crop_pixels:-crop_pixels,
                                                        crop_pixels:-crop_pixels]
            if not DO_BLEND:
                if overlap_x != 0:
                    this_tile_adjusted = this_tile_adjusted[:,
                                                            overlap_x:-overlap_x]
                if overlap_y != 0:
                    this_tile_adjusted = this_tile_adjusted[overlap_y:-overlap_y, :]
            if i % NUM_ROWS == 0:
                overlay_image = this_tile_adjusted
            else:
                if overlap_y == 0 or not DO_BLEND:
                    overlay_image = np.concatenate(
                        (overlay_image, this_tile_adjusted), axis=0)
                else:
                    # blend overlap region of tiles
                    bg_roi = overlay_image[-overlap_y:, :]
                    fg_roi = this_tile_adjusted[:overlap_y, :]
                    roi_h = bg_roi.shape[0]
                    alpha_h = np.linspace(0, 1, roi_h).reshape(-1, 1, 1)
                    blended_roi = (1 - alpha_h) * bg_roi + alpha_h * fg_roi
                    blended_roi = blended_roi.astype(np.uint8)

                    # cv.imshow("Background", overlay_image)
                    # cv.imshow("BG ROI", bg_roi)
                    # cv.imshow("Foreground", this_tile_adjusted)
                    # cv.imshow("FG ROI", fg_roi)
                    # cv.imshow("blended", blended_roi)
                    # cv.waitKey()

                    overlay_image = overlay_image[:-overlap_y, :]
                    this_tile_adjusted = this_tile_adjusted[overlap_y:, :]

                    overlay_image = np.concatenate(
                        (overlay_image, blended_roi, this_tile_adjusted), axis=0)

            if (i+1) % NUM_ROWS == 0:
                # cv.imshow(f"Row {i // NUM_ROWS}", overlay_image)
                # cv.waitKey()
                if final_image is None:
                    final_image = overlay_image
                else:
                    if overlap_x == 0 or not DO_BLEND:
                        final_image = np.concatenate(
                            (final_image, overlay_image), axis=1)
                    else:
                        # blend overlap region of columns
                        bg_roi = final_image[:, -overlap_x:]
                        fg_roi = overlay_image[:, :overlap_x]
                        roi_w = bg_roi.shape[1]
                        alpha_w = np.linspace(0, 1, roi_w).reshape(1, -1, 1)
                        blended_roi = (1 - alpha_w) * bg_roi + alpha_w * fg_roi
                        blended_roi = blended_roi.astype(np.uint8)

                        # cv.imshow("Background", final_image)
                        # cv.imshow("BG ROI", bg_roi)
                        # cv.imshow("Foreground", overlay_image)
                        # cv.imshow("FG ROI", fg_roi)
                        # cv.imshow("blended", blended_roi)
                        # cv.waitKey()

                        final_image = final_image[:, :-overlap_x]
                        overlay_image = overlay_image[:, overlap_x:]

                        final_image = np.concatenate(
                            (final_image, blended_roi, overlay_image), axis=1)

        self.progress.setRange(0, 0)  # indeterminate

        if self.running:  # only if not stopped

            (h, w) = final_image.shape[:2]
            show_factor = 850 / (max(w, h))
            scaled_image = cv.resize(final_image, (0, 0),
                                     fx=show_factor, fy=show_factor)

            if quick:
                self.output_size.setText("Quick Load!")

                # cv.imshow("Stitched", final_image)
                self.set_image(scaled_image)
            else:
                self.output_size.setText(f"Output Size = {w} x {h}")

                # cv.imshow("Stitched", scaled_image)
                self.set_image(scaled_image)

                self.cached_image = final_image
        # output_path = os.path.join(self.image_path, "stitched_output.jpg")
        # cv.imwrite(output_path, final_image)

        # if is_tile:
        #     roi_h = bg_roi.shape[:2][0]
        #     alpha_h = np.linspace(0, 1, roi_h).reshape(-1, 1, 1)
        #     blended_roi = (1 - alpha_h) * bg_roi + alpha_h * fg_roi
        # elif is_column:
        #     roi_w = bg_roi.shape[:2][1]
        #     alpha_w = np.linspace(0, 1, roi_w).reshape(1, -1, 1)
        #     blended_roi = (1 - alpha_w) * bg_roi + alpha_w * fg_roi

        self.setToolbarEnabled(True)

        # hide progress
        self.progress.setVisible(False)
        self.output_size.setVisible(True)

    def setToolbarEnabled(self, enabled, allow_stop=False):
        self.running = False if enabled else True
        b_update = None
        button_found = False
        for i in range(self.toolbar.count()):
            widget = self.toolbar.itemAt(i).widget()
            if widget is not None:
                if widget.text() in ["Update", "Stop"]:
                    b_update = widget
                    button_found = True
                widget.setEnabled(enabled)
        if button_found and allow_stop and not enabled:
            b_update.setText("Stop")
            b_update.setEnabled(True)
        else:
            b_update.setText("Update")

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

        # Option A: Use MEAN value to standardize image brightness
        # vig_mean_val = cv.mean(vig_norm)[0]
        # inv_norm_vig = vig_mean_val / vig_norm  # Compute G = m/F

        # Option B: Use MAX value to standardize image brightness
        # For avoiding "false colors" we may use the maximum instead of the mean.
        vig_max_val = vig_norm.max()
        inv_norm_vig = vig_max_val / vig_norm

        # Convert inv_norm_vig to 3 channels before using cv2.multiply. https://stackoverflow.com/a/48338932/4926757
        self.df_normalizer = cv.cvtColor(inv_norm_vig, cv.COLOR_GRAY2BGR)

    def df_correct(self, image):
        if not CORRECT_DF:
            return image

        # Compute: C = R * G
        return cv.multiply(image, self.df_normalizer, dtype=cv.CV_8U)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = WindowStitcher()
    # path = os.path.dirname(os.getcwd())
    # path = os.path.dirname(path)
    # path = os.path.join(path, "content\\images\\raw_images\\")
    path = os.path.join(
        r"C:\Users\Alexander J. Ross\Documents\SVN Repos\SensorQC\test_6")
    win.config(path)
    # proc.create_stitched_matrix()
    # proc.place_tiles_absolutely()
    win.show()
    sys.exit(app.exec_())
