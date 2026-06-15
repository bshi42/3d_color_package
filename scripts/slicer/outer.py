import logging
import os
import sys


logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

logger.info("This is an info message.")
print(sys.version)
print(os.cpu_count())

exit()
