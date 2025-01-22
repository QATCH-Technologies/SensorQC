import logging
import multiprocessing
import os
import time
import numpy as np

from logging.handlers import QueueListener
from process_pipeline import ProcessPipeline
from process_stitch import ProcessStitcher
from process_detect import ProcessDetector
from process_common import NUM_TILES, NUM_ROWS
from io.runner import ScanRunner


class Stitcher:
    def __init__(self, image_dir, run_scan_realtime=False, log_levels=None):
        self.image_dir = image_dir
        self.run_scan_realtime = run_scan_realtime
        self.log_levels = log_levels or {
            'main': logging.INFO,
            'pipeline': logging.INFO,
            'stitch': logging.DEBUG,
            'detect': logging.DEBUG
        }

        # Setup logging
        self.logger = self._initialize_logger()
        self.logger.debug("Initializing Stitcher...")

        # Setup queues
        self._setup_queues()

        # Initialize processes
        self.pipeline = ProcessPipeline(
            queue_log=self.queue_log,
            queue_rx1=self.queue_main_to_children,
            queue_tx1A=self.queue_main_to_stitch,
            queue_tx1B=self.queue_main_to_detect,
            log_level=self.log_levels['pipeline']
        )

        self.stitcher = ProcessStitcher(
            queue_log=self.queue_log,
            queue_rx1A=self.queue_main_to_stitch,
            queue_rx1B=self.queue_main_to_detect,
            queue_rx2=self.queue_detect_to_stitch,
            queue_tx3=self.queue_stitch_to_detect,
            log_level=self.log_levels['stitch']
        )

        self.detector = ProcessDetector(
            queue_log=self.queue_log,
            queue_rx1B=self.queue_main_to_detect,
            queue_tx2=self.queue_detect_to_stitch,
            queue_rx3=self.queue_stitch_to_detect,
            log_level=self.log_levels['detect']
        )

    def _initialize_logger(self):
        """Initialize the logger."""
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(self.log_levels['main'])
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

        # Queue listener for multiprocessing logging
        self.queue_log = multiprocessing.Queue()
        self.log_listener = QueueListener(self.queue_log, handler)
        self.log_listener.start()

        return logger

    def _setup_queues(self):
        """Setup multiprocessing queues."""
        self.queue_main_to_children = multiprocessing.Queue()
        self.queue_main_to_stitch = multiprocessing.Queue()
        self.queue_main_to_detect = multiprocessing.Queue()
        self.queue_detect_to_stitch = multiprocessing.Queue()
        self.queue_stitch_to_detect = multiprocessing.Queue()

    def _load_images(self):
        """Load and sort image file paths."""
        if not os.path.exists(self.image_dir):
            self.logger.warning(
                f"Creating missing directory: {os.path.basename(self.image_dir)}")
            os.makedirs(self.image_dir)

        if self.run_scan_realtime:
            image_paths = [f"tile_{i + 1}.jpg" for i in range(NUM_TILES)]
        else:
            image_paths = [
                path for path in os.listdir(self.image_dir)
                if path.endswith("jpg") and not path.startswith("stitched")
            ]

        image_ids = [int(path[path.rindex("_") + 1:-4])
                     for path in image_paths]
        sort_order = np.argsort(image_ids)
        return np.array(image_paths)[sort_order], np.array(image_ids)[sort_order]

    def _calculate_grid(self, total_tiles):
        """Calculate the grid dimensions based on total tiles and rows."""
        grid_rows = NUM_ROWS
        grid_cols = total_tiles / grid_rows
        if grid_cols % 1 != 0:
            raise ValueError(
                "Number of tiles not divisible by rows. Check and try again.")
        return grid_rows, int(grid_cols)

    def run(self):
        """Run the Stitcher pipeline."""
        image_paths, image_ids = self._load_images()
        total_tiles = len(image_ids)
        grid_rows, grid_cols = self._calculate_grid(total_tiles)

        image_positions = [
            {"col": (id - 1) % grid_rows, "row": int((id - 1) / grid_rows)}
            for id in image_ids
        ]

        # Configure and start processes
        processes = ['pipeline', 'detector', 'stitcher']
        for proc_name in processes:
            proc = getattr(self, proc_name)
            proc.config(self.image_dir)
            proc.start()

        for col in range(grid_cols):
            if self.run_scan_realtime:

                self._wait_for_realtime_tiles(col, grid_rows, total_tiles)
            elif col > 0:
                time.sleep(5)
            this_row = [
                image_paths[idx] for idx, pos in enumerate(image_positions) if pos["col"] == col
            ]

            self.logger.info(f"Passing in tiles for column {col + 1}")
            self.queue_main_to_children.put(this_row)

        self.queue_main_to_children.put(None)  # Sentinel value

        for proc_name in processes:
            self.logger.debug(f"Waiting on {proc_name} to finish...")
            proc = getattr(self, proc_name)
            proc.join()

        self.logger.debug("Shutting down logger...")
        self.log_listener.stop()

    def _wait_for_realtime_tiles(self, col, grid_rows, total_tiles):
        """Wait for real-time tiles to be scanned."""
        self.logger.info("Waiting for scanned images to be read...")
        num_tiles, last_num_tiles = -1, -1

        while num_tiles < (col + 1) * grid_rows:
            num_tiles = len([
                path for path in os.listdir(self.image_dir)
                if path.endswith("jpg") and not path.startswith("stitched")
            ])
            if num_tiles != last_num_tiles:
                self.logger.debug(f"Found {num_tiles} tiles...")
            if col == 0 and num_tiles == total_tiles:
                raise Exception(
                    "Running in real-time mode but all the tiles already exist! "
                    "Pick a new path or delete JPGs and try again."
                )
            last_num_tiles = num_tiles


if __name__ == "__main__":
    IMAGE_DIR = os.path.join(
        os.getcwd(), "../../content/images/bf_c_raw/bf_c_raw")
    stitcher = Stitcher(image_dir=IMAGE_DIR, run_scan_realtime=False)
    stitcher.run()
