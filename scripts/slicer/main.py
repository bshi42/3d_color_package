import logging
import os
import sys
import slicer
from pathlib import Path

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(sys.stdout))

def main():
    
    slicer.util.pip_install("opencv-python")

    import cv2
    print("OpenCV version:", cv2.__version__)

    exit()

if __name__ == "__main__":
    main()
