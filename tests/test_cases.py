# importing testing, arrays, opencv, and system stuff
import unittest
import numpy as np
import cv2
import sys
import os

# adding the main project folder so the test file can import from src
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            '..'
        )
    )
)

# importing the motion detector class
from src.motion_detector import MotionDetector

# making a test class for motion detector
class TestMotionDetector(unittest.TestCase):
    
    def setUp(self):

        # creating a fresh detector before every test
        self.detector = MotionDetector()

    def test_no_movement(self):

        # feeding empty black frames first
        # this helps the background model learn the scene
        for _ in range(5):

            frame = np.zeros((100, 100, 3), dtype=np.uint8)

            self.detector.detect_motion(frame)
            
        # another empty frame with no movement
        current_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # checking detections
        boxes, _ = self.detector.detect_motion(current_frame)

        # expecting zero detections
        self.assertEqual(len(boxes), 0)

    def test_identifies_significant_movement(self):

        # first initializing the background
        for _ in range(10):

            self.detector.detect_motion(
                np.zeros((100, 100, 3), dtype=np.uint8)
            )
            
        # creating a fake moving object
        current_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # drawing a tall white rectangle like a human shape
        cv2.rectangle(
            current_frame,
            (30, 10),
            (50, 90),
            (255, 255, 255),
            -1
        )
        
        # running detection twice
        # this is needed because the detector checks stability between frames
        self.detector.detect_motion(current_frame)

        boxes, _ = self.detector.detect_motion(current_frame)
        
        # checking if motion was detected
        self.assertGreater(
            len(boxes),
            0,
            "Significant motion should be detected"
        )

    def test_ignores_small_noise(self):

        # making an empty frame
        current_frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # drawing a tiny white square
        # this should be treated as noise
        cv2.rectangle(
            current_frame,
            (10, 10),
            (12, 12),
            (255, 255, 255),
            -1
        )
        
        # checking detections
        boxes, _ = self.detector.detect_motion(current_frame)

        # expecting zero detections because the object is too small
        self.assertEqual(
            len(boxes),
            0,
            "Small noise should be filtered out"
        )

# starting the test runner
if __name__ == '__main__':

    # running all tests
    unittest.main()