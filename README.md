# Arrow Detection Rover Control

This repository contains Python code for detecting arrows using OpenCV, and subsequently controlling a rover to move in the direction pointed by the arrow. The project employs computer vision techniques, including edge detection, contour detection, and template matching to identify arrow directions in a video feed.

## Table of Contents
1. [Introduction](#introduction)
2. [Dependencies](#dependencies)
3. [Setup and Installation](#setup-and-installation)
4. [Usage](#usage)
5. [Code Overview](#code-overview)
    - [Edge Detection](#edge-detection)
    - [Grayscale and Blur](#grayscale-and-blur)
    - [Contour Detection](#contour-detection)
    - [Arrow Tip Identification](#arrow-tip-identification)
    - [Direction Determination](#direction-determination)
    - [Template Matching](#template-matching)
    - [Frame Processing](#frame-processing)
7. [Bibliography](#bibliography)

## Introduction
This project demonstrates how to use OpenCV for real-time arrow detection in a video feed and control a rover based on the detected arrow direction. The code processes each frame to detect arrows pointing left or right, and commands the rover accordingly.

## Dependencies
- Python 3.10.12
- OpenCV
- NumPy

## Setup and Installation
Install the Dependencies on your computer and then proceed to download the template

The right and left arrow images are loaded in grayscale:
Two images of arrow for pointing in both the direction are used as a template.
These images will be used later for template matching.

Press 'q' to quit the video feed.

## Code Overview
The main components of the code are as follows:

### Edge Detection
Algorithm Used: Canny Edge Detection

Explanation:
Edge detection is a fundamental image processing technique that identifies points in an image where the brightness changes sharply. The Canny edge detection algorithm is popular for its effectiveness and involves the following steps:

Noise Reduction: Apply a Gaussian blur to smooth the image and reduce noise.
Gradient Calculation: Compute the intensity gradient of the image using derivative filters.
Non-Maximum Suppression: Thin out the edges to create a single-pixel-wide line for better accuracy.
Double Threshold: Classify edges into strong, weak, and non-edges based on two threshold values.
Edge Tracking by Hysteresis: Finalize the edge detection by tracking edges starting from strong edges and including weak edges that are connected to strong edges.

### Morphological Transformations
Algorithm Used: Morphological Closing

Explanation:
Morphological transformations are operations that process images based on their shapes. They apply a structuring element to an input image and output an image of the same size. Morphological closing is specifically used to fill small holes in objects:

Dilation: Increases the white region in the image or the size of the foreground object.
Erosion: Reduces the white region in the image or the size of the foreground object.
Closing: A dilation operation followed by an erosion. It helps to close small gaps in edges and is useful in strengthening the detected edges.

### Grayscale Conversion and Blurring
Algorithm Used: Gaussian Blur

Explanation:
Grayscale conversion simplifies the image by removing color information, retaining only the intensity of light. Gaussian blur is then applied to reduce image noise and detail, which is helpful in preprocessing for further image analysis. It uses a Gaussian function to create a convolution kernel, resulting in a smoothing effect on the image.

### Contour Detection
Algorithm Used: Contour Detection via cv2.findContours

Explanation:
Contour detection is used to identify the boundaries of objects in an image:

Binary Image: Contour detection requires a binary image (edges detected or thresholded image).
Hierarchy and Retrieval Modes: Determines how contours are retrieved and organized.
Contour Approximation: Reduces the number of points in a contour for simpler shape representation. This is achieved using the Ramer-Douglas-Peucker algorithm.

### Convex Hull
Algorithm Used: Convex Hull Calculation

Explanation:
The convex hull of a shape is the smallest convex shape that can entirely contain it. For a set of points, the convex hull is the smallest polygon that encloses all the points. It is used in the project to help identify the arrow tip by understanding the geometric structure of the arrow.

###Template Matching
Algorithm Used: Template Matching via cv2.matchTemplate

Explanation:
Template matching is used to find parts of an image that match a template image:

Sliding Window: The template image slides over the input image, and at each position, a similarity metric (like correlation or squared differences) is computed.
Best Match: The location where the highest similarity metric is found is considered the best match. Multiple scales of the template are used to handle size variations in the arrows.

### Direction Determination
Algorithm Used: Geometric Analysis of Contours

Explanation:
To determine the direction of the arrow:

Identifying the Arrow Tip: Use the geometry of the contour and the convex hull to find the tip of the arrow.
Point Distribution: Compare the number of points on the left and right sides of the arrow tip. If significantly more points are on one side, the arrow is pointing in the opposite direction.


## Bibliography
- OpenCV documentation: https://docs.opencv.org/
- NumPy documentation: https://numpy.org/doc/stable/
