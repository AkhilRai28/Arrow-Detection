import unittest
import cv2
import numpy as np
from src.arrow_recognition import edge_detection

class TestEdgeDetection(unittest.TestCase):
    def test_edge_detection(self):
        image = cv2.imread("Right_Arrow.jpg", cv2.IMREAD_GRAYSCALE)
        edges = edge_detection(image)
        self.assertIsNotNone(edges)
        self.assertEqual(edges.shape, image.shape)

if __name__ == '__main__':
    unittest.main()
