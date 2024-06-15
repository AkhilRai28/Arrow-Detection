import cv2
import numpy as np
import math

right_arrow = cv2.imread("Right_Arrow.jpg", cv2.IMREAD_GRAYSCALE)
left_arrow = cv2.imread("Left_Arrow.jpg", cv2.IMREAD_GRAYSCALE)
MATCH_THRESHOLD = 0.8

def edge_detection(image):
    edges = cv2.Canny(image, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    return edges

def to_grayscale_and_blur(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    return blurred

def detect_contours(image):
    processed = edge_detection(to_grayscale_and_blur(image))
    contours, _ = cv2.findContours(processed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def identify_arrow_tip(points, hull_indices):
    remaining_indices = np.setdiff1d(np.arange(len(points)), hull_indices)
    for i in range(2):
        j = (remaining_indices[i] + 2) % len(points)
        if np.array_equal(points[j], points[remaining_indices[i-1] - 2]):
            return tuple(points[j])
    return None

def determine_direction(approx, tip):
    left_points = sum(1 for pt in approx if pt[0][0] > tip[0])
    right_points = sum(1 for pt in approx if pt[0][0] < tip[0])
    
    if left_points > right_points and left_points > 4:
        return "Left"
    elif right_points > left_points and right_points > 4:
        return "Right"
    return "None"

def template_matching(image, template):
    best_match = {"value": -1, "location": -1, "scale": -1}
    for scale in np.linspace(0.1, 0.5, 15):
        resized_template = cv2.resize(template, None, fx=scale, fy=scale)
        match_result = cv2.matchTemplate(image, resized_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(match_result)
        if max_val > best_match["value"] and max_val > MATCH_THRESHOLD:
            best_match.update({"value": max_val, "location": max_loc, "scale": scale})
    return best_match

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

def match_and_annotate(frame, template, color, label):
    gray_frame = to_grayscale_and_blur(frame)
    match = template_matching(gray_frame, template)
    
    if match["location"] != -1:
        w, h = int(template.shape[1] * match["scale"]), int(template.shape[0] * match["scale"])
        top_left = match["location"]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        angle = math.degrees(math.atan2(top_left[1] - (frame.shape[0] // 2), top_left[0] - (frame.shape[1] // 2)))
        print(f'{label} arrow detected at angle: {angle}')

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

if __name__ == "__main__":
    main()