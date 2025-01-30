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

LOG_LEVEL_MAIN = logging.INFO
LOG_LEVEL_PIPE = logging.INFO
LOG_LEVEL_STITCH = logging.DEBUG
LOG_LEVEL_DETECT = logging.DEBUG

RUN_SCAN_REALTIME = False


class StitcherTest:

    def __init__(self):

        # Instantiate multiprocessing logger
        h = logging.StreamHandler()
        f = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        h.setFormatter(f)
        self.__log__(h)

        self.logger.debug("Main Process initializing...")

        # Queue 0: Logging queue used to handle logging messages from all child processes
        # Queue 1: Passes a list of tiles as rows are captured to stitcher & detector processes
        # Queue 1A: Sub-queue of Queue 1 for handling commands from main to stitcher process
        # Queue 1B: Sub-queue of Queue 1 for handling commands from main to detector process
        # Queue 2: Passes calculated tile offsets from detector to stitcher process
        # Queue 3: Passes calculated avg/std offsets from stitcher to detector process
        self.queue0_logger = multiprocessing.Queue()
        self.queue1_main_to_children = multiprocessing.Queue()
        self._queue1A_main_to_stitch = multiprocessing.Queue()
        self._queue1B_main_to_detect = multiprocessing.Queue()
        self.queue2_detect_to_stitch = multiprocessing.Queue()
        self.queue3_stitch_to_detect = multiprocessing.Queue()

        # Start handling logger messages from children processes
        self.log_listener = QueueListener(self.queue0_logger, h)
        self.log_listener.start()

        # Process 0: Pipeline - splits queues to children (Queue 1 --> 1A & 1B)
        self.pipeline = ProcessPipeline(
            queue_log=self.queue0_logger,
            queue_rx1=self.queue1_main_to_children,
            queue_tx1A=self._queue1A_main_to_stitch,
            queue_tx1B=self._queue1B_main_to_detect,
            log_level=LOG_LEVEL_PIPE,
        )

        # Process 1: Stitcher - combines tiles
        self.stitcher = ProcessStitcher(
            queue_log=self.queue0_logger,
            queue_rx1A=self._queue1A_main_to_stitch,
            queue_rx1B=self._queue1B_main_to_detect,
            queue_rx2=self.queue2_detect_to_stitch,
            queue_tx3=self.queue3_stitch_to_detect,
            log_level=LOG_LEVEL_STITCH,
        )

        # Process 2: Detector - detects offsets
        self.detector = ProcessDetector(
            queue_log=self.queue0_logger,
            queue_rx1B=self._queue1B_main_to_detect,
            queue_tx2=self.queue2_detect_to_stitch,
            queue_rx3=self.queue3_stitch_to_detect,
            log_level=LOG_LEVEL_DETECT,
        )

        self.logger.debug("Main Process initialized")

    def __log__(self, h):

        self.logger = logging.getLogger(__class__.__name__)
        self.logger.addHandler(h)
        self.logger.setLevel(LOG_LEVEL_MAIN)

    def run(self):

        # Load the images
        image_path = os.path.join(
            os.getcwd(), "../../content/images/df_c")
        if not os.path.exists(image_path):
            self.logger.warning(
                f"Creating missing directory: {os.path.basename(image_path)}")
            os.makedirs(image_path)

        if RUN_SCAN_REALTIME:
            self.logger.info(f"Image path is: {image_path}")
            image_paths = [f"tile_{i+1}.jpg" for i in range(NUM_TILES)]
        else:
            image_paths = [path for path in os.listdir(image_path) if path.endswith(
                "jpg") and not path.startswith("stitched")]
        image_ids = [int(path[path.rindex("_")+1:-4]) for path in image_paths]
        sort_order = np.argsort(image_ids)
        sorted_paths = np.array(image_paths)[sort_order]
        image_ids = np.array(image_ids)[sort_order]

        # TODO calculate 'grid_rows' automatically, somehow
        total_tiles = len(image_ids)
        grid_rows = NUM_ROWS  # vertical tile count, Y-axis
        grid_cols = total_tiles / grid_rows  # horizontal tile count, X-axis
        assert grid_cols % 1 == 0, "Number of tiles not divisible by 'rows'. Check and try again."
        grid_cols = int(grid_cols)

        image_row = [(id - 1) % grid_rows for id in image_ids]
        image_col = [int((id - 1) / grid_rows) for id in image_ids]

        image_pos = []
        for i in range(len(image_ids)):
            image_pos.append({"col": image_col[i], "row": image_row[i]})

        # self.logger.warning('This is a warning')
        # self.logger.error('This is an error')

        # Configure and start other processes, they will idle until data is queued
        processes = ['pipeline', 'detector', 'stitcher']
        for pn in processes:
            proc = getattr(self, pn)
            proc.config(image_path)
            proc.start()

        # self.pipeline.start()
        # self.detector.start()
        # self.stitcher.start()

        for a in range(grid_cols):
            # if a == 5: break

            if RUN_SCAN_REALTIME:
                # get number of tiles scanned so far and wait until we have a full row to process
                self.logger.info("Waiting for scanned images to be read...")
                num_tiles, last_num_tiles = -1, -1
                while num_tiles < (a + 1) * grid_rows:
                    num_tiles = len([path for path in os.listdir(image_path) if path.endswith(
                        "jpg") and not path.startswith("stitched")])
                    if num_tiles != last_num_tiles:
                        self.logger.debug(f"Found {num_tiles} tiles...")
                    if a == 0 and num_tiles == total_tiles:
                        raise Exception(
                            "Running in real-time mode but all the tiles already exist! " +
                            "Pick a new path or delete JPGs and try again.")
                    last_num_tiles = num_tiles
            elif a > 0:
                time.sleep(5)

            this_row = []
            for b in range(total_tiles):
                if image_pos[b]["col"] == a:
                    this_row.append(sorted_paths[b])
            # this_row = list(this_row[::-1]) # reverse, work bottom to top

            self.logger.info(f"Passing in tiles for column {a+1}")
            self.logger.debug(f"Enqueueing: {this_row} (len={len(this_row)})")
            self.queue1_main_to_children.put(this_row)

        # Queueing sentinel value indicates end of processing
        self.queue1_main_to_children.put(None)

        # Wait for processes to finish
        for pn in processes:
            self.logger.debug(f"Waiting on {pn} to finish...")
            proc = getattr(self, pn)
            proc.join()

        # Purge log listener queue and stop thread
        self.logger.debug("Waiting on logger to finish...")
        self.log_listener.stop()

        self.logger.debug("All tasks finished, closing...")


if __name__ == "__main__":
    StitcherTest().run()
