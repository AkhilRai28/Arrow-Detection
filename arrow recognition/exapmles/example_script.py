# examples/example_script.py
import cv2
from src.arrow_recognition import process_frame, match_and_annotate

right_arrow = cv2.imread("data/Right_Arrow.jpg", cv2.IMREAD_GRAYSCALE)
left_arrow = cv2.imread("data/Left_Arrow.jpg", cv2.IMREAD_GRAYSCALE)

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
