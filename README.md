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
    - [Main Loop](#main-loop)
6. [Repository Name and Tags](#repository-name-and-tags)
7. [Bibliography](#bibliography)

## Introduction
This project demonstrates how to use OpenCV for real-time arrow detection in a video feed and control a rover based on the detected arrow direction. The code processes each frame to detect arrows pointing left or right, and commands the rover accordingly.

## Dependencies
- Python 3.x
- OpenCV
- NumPy

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/arrow-detection-rover-control.git
   ```
2. Install the required dependencies:
   ```bash
   pip install opencv-python-headless numpy
   ```

3. Place the right and left arrow images in the specified directory or update the paths in the code:
   ```python
   right_arrow = cv2.imread("/path/to/right.jpg", cv2.IMREAD_GRAYSCALE)
   left_arrow = cv2.imread("/path/to/left.jpg", cv2.IMREAD_GRAYSCALE)
   ```

## Usage
Run the main script to start the video feed and arrow detection:
```bash
python main.py
```
Press 'q' to quit the video feed.

## Code Overview
The main components of the code are as follows:

### Edge Detection
The `edge_detection` function applies the Canny edge detection algorithm and morphological transformations to the input image to highlight edges.
```python
def edge_detection(image):
    edges = cv2.Canny(image, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edges
```

### Grayscale and Blur
The `to_grayscale_and_blur` function converts the input image to grayscale and applies Gaussian blur to reduce noise.
```python
def to_grayscale_and_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    return blurred
```

### Contour Detection
The `detect_contours` function processes the image to detect contours.
```python
def detect_contours(image):
    processed = edge_detection(to_grayscale_and_blur(image))
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours
```

### Arrow Tip Identification
The `identify_arrow_tip` function identifies the tip of the arrow from the contour points and convex hull indices.
```python
def identify_arrow_tip(points, hull_indices):
    remaining_indices = np.setdiff1d(np.arange(len(points)), hull_indices)
    for i in range(2):
        j = (remaining_indices[i] + 2) % len(points)
        if np.array_equal(points[j], points[remaining_indices[i-1] - 2]):
            return tuple(points[j])
    return None
```

### Direction Determination
The `determine_direction` function determines the direction of the arrow based on the identified tip.
```python
def determine_direction(approx, tip):
    left_points = sum(1 for pt in approx if pt[0][0] > tip[0])
    right_points = sum(1 for pt in approx if pt[0][0] < tip[0])
    
    if left_points > right_points and left_points > 4:
        return "Left"
    elif right_points > left_points and right_points > 4:
        return "Right"
    return "None"
```

### Template Matching
The `template_matching` function performs template matching to identify arrows in the frame.
```python
def template_matching(image, template):
    best_match = {"value": -1, "location": -1, "scale": -1}
    for scale in np.linspace(0.1, 0.5, 15):
        resized_template = cv2.resize(template, None, fx=scale, fy=scale)
        match_result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(match_result)
        if max_val > best_match["value"] and max_val > MATCH_THRESHOLD:
            best_match.update({"value": max_val, "location": max_loc, "scale": scale})
    return best_match
```

### Frame Processing
The `process_frame` function processes each frame to detect contours and identify arrow directions.
```python
def process_frame(frame):
    contours = detect_contours(frame)
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        hull = cv2.convexHull(approx, returnPoints=False)
        
        if 4 < len(hull) < 6 and len(hull) + 2 == len(approx) and len(approx) > 6:
            tip = identify_arrow_tip(approx[:, 0, :], hull.squeeze())
            if tip:
                direction = determine_direction(approx, tip)
                if direction != "None":
                    cv2.drawContours(frame, [contour], -1, (0, 255, 0), 3)
                    cv2.circle(frame, tip, 3, (0, 0, 255), -1)
                    print('Arrow Direction:', direction)

    return frame
```

### Main Loop
The `main` function captures video frames from the webcam and processes each frame to detect and annotate arrows.
```python
def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame)
        match_and_annotate(processed_frame, right_arrow, (0, 255, 0), 'Right')
        match_and_annotate(processed_frame, left_arrow, (255, 0, 0), 'Left')
        
        cv2.imshow("Video Feed", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
```

## Repository Name and Tags
**Repository Name:** arrow-detection-rover-control

**Tags:** 
- OpenCV
- Computer Vision
- Python
- Arrow Detection
- Robotics
- Template Matching
- Edge Detection
- Contour Detection

## Bibliography
- OpenCV documentation: https://docs.opencv.org/
- NumPy documentation: https://numpy.org/doc/stable/

Feel free to modify the code to suit your specific requirements and extend the functionality as needed. For any issues or contributions, please open a new issue or pull request on GitHub.
