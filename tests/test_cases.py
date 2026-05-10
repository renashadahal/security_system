import unittest
import numpy as np
import cv2
from src.motion_detector import MotionDetector

class TestMotionDetector(unittest.TestCase):
    
    def setUp(self):
        # create a 100x100 black base frame
        self.base_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        self.detector = MotionDetector(self.base_frame)

    def test_no_movement(self):
        # test with an identical black frame
        current_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        boxes, _ = self.detector.detect_motion(current_frame)
        # assert that no bounding boxes were found
        self.assertEqual(len(boxes), 0)

    def test_identifies_significant_movement(self):
        # create a frame with a large white square (simulating a person)
        current_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(current_frame, (20, 20), (60, 60), (255, 255, 255), -1)
        
        boxes, _ = self.detector.detect_motion(current_frame)
        # assert that movement was detected
        self.assertGreater(len(boxes), 0)

    def test_ignores_small_noise(self):
        # create a frame with a tiny white dot (simulating sensor noise)
        current_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(current_frame, (10, 10), (12, 12), (255, 255, 255), -1)
        
        boxes, _ = self.detector.detect_motion(current_frame)
        # assert that the 500-pixel area filter ignored this noise
        self.assertEqual(len(boxes), 0)

if __name__ == '__main__':
    unittest.main()