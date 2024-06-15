### Introduction
This code is designed to detect arrows in a video feed using OpenCV and Python, and determine the direction they are pointing. Based on the detected direction, the code can be used to control a rover to move accordingly.

### Dependencies
The code uses the following Python libraries:
- **OpenCV**: For image processing tasks.
- **NumPy**: For numerical operations, particularly useful for handling arrays and matrices.

### Loading Arrow Images
The right and left arrow images are loaded in grayscale:
```python
right_arrow = cv2.imread("/home/yagna/Code/Rover/Images/right.jpg", cv2.IMREAD_GRAYSCALE)
left_arrow = cv2.imread("/home/yagna/Code/Rover/Images/left.jpg", cv2.IMREAD_GRAYSCALE)
```
These images will be used later for template matching.

### Edge Detection
The `edge_detection` function applies the Canny edge detection algorithm and then performs morphological closing to strengthen the edges:
```python
def edge_detection(image):
    edges = cv2.Canny(image, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edges
```
- **Canny Edge Detection**: Detects edges in the image.
- **Morphological Closing**: Fills small gaps in the detected edges.

### Grayscale and Blur
The `to_grayscale_and_blur` function converts an image to grayscale and then applies Gaussian blur:
```python
def to_grayscale_and_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    return blurred
```
- **Grayscale Conversion**: Simplifies the image by removing color information.
- **Gaussian Blur**: Reduces noise and detail in the image.

### Contour Detection
The `detect_contours` function processes the image to find contours:
```python
def detect_contours(image):
    processed = edge_detection(to_grayscale_and_blur(image))
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours
```
- **Contour Detection**: Identifies the outlines of objects in the processed image.

### Identifying Arrow Tip
The `identify_arrow_tip` function identifies the tip of the arrow from the contour points and convex hull indices:
```python
def identify_arrow_tip(points, hull_indices):
    remaining_indices = np.setdiff1d(np.arange(len(points)), hull_indices)
    for i in range(2):
        j = (remaining_indices[i] + 2) % len(points)
        if np.array_equal(points[j], points[remaining_indices[i-1] - 2]):
            return tuple(points[j])
    return None
```
- **Convex Hull**: The smallest convex shape that can enclose all the points.
- **Arrow Tip Identification**: Finds the point that is likely the tip of the arrow based on the geometry of the contour.

### Determining Arrow Direction
The `determine_direction` function determines the direction of the arrow based on the identified tip:
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
- **Direction Determination**: Compares the number of points on the left and right side of the tip to determine the direction.

### Template Matching
The `template_matching` function performs template matching to find arrows in the image:
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
- **Template Matching**: Searches for the best match of the template in the image at various scales.

### Frame Processing
The `process_frame` function processes each frame to detect contours and identify arrow directions:
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
- **Contour Approximation**: Simplifies contours to reduce the number of points.
- **Hull and Tip Identification**: Checks the shape to identify potential arrows and their tips.
- **Drawing and Annotation**: Draws contours and marks the arrow tip on the frame.

### Main Function
The `main` function captures video frames from the webcam and processes each frame to detect and annotate arrows:
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
- **Video Capture**: Uses the webcam to capture frames.
- **Frame Processing**: Each frame is processed to detect arrows.
- **Template Matching and Annotation**: Detects arrows using template matching and annotates them on the frame.
- **Displaying the Video**: Shows the processed video feed with annotations.
- **Exit Condition**: Press 'q' to exit the video feed.

### Conclusion
This code provides a comprehensive solution for detecting arrow directions using computer vision techniques. By combining edge detection, contour analysis, and template matching, the system can accurately identify and annotate arrows in real-time video feeds.
