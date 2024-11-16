import logging
import multiprocessing

from logging.handlers import QueueHandler

class ProcessPipeline(multiprocessing.Process):

    def __init__(self, queue_log, queue_rx1, queue_tx1A, queue_tx1B,
                 log_level = logging.DEBUG):
        multiprocessing.Process.__init__(self)

        self.queue_log = queue_log
        self.log_level = log_level
        self.__log__()

        self.logger.debug("Pipeline process initializing...")

        self.queue_rx1 = queue_rx1
        self.queue_tx1A = queue_tx1A
        self.queue_tx1B = queue_tx1B

        self.logger.debug("Pipeline process initialized")

    
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

        while True:
            item = self.queue_rx1.get()

            # Forward item to child queues (always)
            self.logger.debug(f"Forward: {item}")
            self.queue_tx1A.put(item)
            self.queue_tx1B.put(item)

            if item is None: # Sentinel value to signal end of processing
                break

        # self.logger.debug("Waiting on children to finish...")

        # while True:
        #     if self.queue_tx1A.empty() and self.queue_tx1B.empty():
        #         break

        self.logger.debug("Finished!")
