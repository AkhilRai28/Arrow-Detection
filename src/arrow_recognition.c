#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

const double MATCH_THRESHOLD = 0.8;

//start
void log_initialization();
void log_template_loaded(const char* template_name);
void log_edge_detection();
void log_contour_detection();
void log_template_matching();
void process_frame(Mat frame);
void match_and_annotate(Mat frame, Mat template_img, Scalar color, const char* label);
Mat edge_detection(Mat image);
Mat to_grayscale_and_blur(Mat image);
vector<vector<Point>> detect_contours(Mat image);
Point identify_arrow_tip(vector<Point> points, vector<int> hull_indices);
const char* determine_direction(vector<Point> approx, Point tip);
double calculate_angle(Point p1, Point p2);
VideoCapture init_video_capture();

int main() {
    log_initialization();
    Mat right_arrow = imread("Right_Arrow.jpg", IMREAD_GRAYSCALE);
    Mat left_arrow = imread("Left_Arrow.jpg", IMREAD_GRAYSCALE);
    if (right_arrow.empty() || left_arrow.empty()) {
        printf("Error loading template images\n");
        return -1;
    }

    log_template_loaded("Right_Arrow");
    log_template_loaded("Left_Arrow");

    VideoCapture cap = init_video_capture();
    if (!cap.isOpened()) {
        printf("Error opening video capture\n");
        return -1;
    }

    Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) {
            printf("Error capturing frame\n");
            break;
        }

        process_frame(frame);
        match_and_annotate(frame, right_arrow, Scalar(0, 255, 0), "Right");
        match_and_annotate(frame, left_arrow, Scalar(255, 0, 0), "Left");

        imshow("Video Feed", frame);

        if (waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

void log_initialization() {
    printf("System initialization...\n");
}

void log_template_loaded(const char* template_name) {
    printf("Template %s loaded.\n", template_name);
}

void log_edge_detection() {
    printf("Performing edge detection...\n");
}

void log_contour_detection() {
    printf("Detecting contours...\n");
}

void log_template_matching() {
    printf("Performing template matching...\n");
}

Mat edge_detection(Mat image) {
    log_edge_detection();
    Mat edges;
    Canny(image, edges, 50, 150);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(edges, edges, MORPH_CLOSE, kernel, Point(-1, -1), 2);
    return edges;
}

Mat to_grayscale_and_blur(Mat image) {
    Mat gray, blurred;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    GaussianBlur(gray, blurred, Size(7, 7), 0);
    return blurred;
}

vector<vector<Point>> detect_contours(Mat image) {
    log_contour_detection();
    Mat processed = edge_detection(to_grayscale_and_blur(image));
    vector<vector<Point>> contours;
    findContours(processed, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
    return contours;
}

Point identify_arrow_tip(vector<Point> points, vector<int> hull_indices) {
    vector<int> remaining_indices;
    for (size_t i = 0; i < points.size(); ++i) {
        if (find(hull_indices.begin(), hull_indices.end(), i) == hull_indices.end()) {
            remaining_indices.push_back(i);
        }
    }

    for (size_t i = 0; i < 2; ++i) {
        size_t j = (remaining_indices[i] + 2) % points.size();
        if (points[j] == points[(remaining_indices[i - 1] - 2 + points.size()) % points.size()]) {
            return points[j];
        }
    }
    return Point(-1, -1);
}

const char* determine_direction(vector<Point> approx, Point tip) {
    int left_points = 0, right_points = 0;
    for (auto& pt : approx) {
        if (pt.x > tip.x) left_points++;
        else if (pt.x < tip.x) right_points++;
    }

    if (left_points > right_points && left_points > 4) return "Left";
    if (right_points > left_points && right_points > 4) return "Right";
    return "None";
}

double calculate_angle(Point p1, Point p2) {
    return atan2(p1.y - p2.y, p1.x - p2.x) * 180.0 / CV_PI;
}

void process_frame(Mat frame) {
    vector<vector<Point>> contours = detect_contours(frame);
    for (auto& contour : contours) {
        vector<Point> approx;
        approxPolyDP(contour, approx, 0.02 * arcLength(contour, true), true);
        vector<int> hull;
        convexHull(Mat(approx), hull, false);

        if (hull.size() > 4 && hull.size() < 6 && hull.size() + 2 == approx.size() && approx.size() > 6) {
            Point tip = identify_arrow_tip(approx, hull);
            if (tip.x != -1 && tip.y != -1) {
                const char* direction = determine_direction(approx, tip);
                if (strcmp(direction, "None") != 0) {
                    drawContours(frame, vector<vector<Point>>{contour}, -1, Scalar(0, 255, 0), 3);
                    circle(frame, tip, 3, Scalar(0, 0, 255), -1);
                    printf("Arrow Direction: %s\n", direction);
                }
            }
        }
    }
}

void match_and_annotate(Mat frame, Mat template_img, Scalar color, const char* label) {
    log_template_matching();
    Mat gray_frame = to_grayscale_and_blur(frame);
    double best_value = -1;
    Point best_location = Point(-1, -1);
    double best_scale = -1;

    for (double scale = 0.1; scale <= 0.5; scale += 0.027) {
        Mat resized_template;
        resize(template_img, resized_template, Size(), scale, scale);
        Mat result;
        matchTemplate(gray_frame, resized_template, result, TM_CCOEFF_NORMED);
        double min_val, max_val;
        Point min_loc, max_loc;
        minMaxLoc(result, &min_val, &max_val, &min_loc, &max_loc);

        if (max_val > best_value && max_val > MATCH_THRESHOLD) {
            best_value = max_val;
            best_location = max_loc;
            best_scale = scale;
        }
    }

    if (best_location.x != -1 && best_location.y != -1) {
        int w = int(template_img.cols * best_scale);
        int h = int(template_img.rows * best_scale);
        Point top_left = best_location;
        Point bottom_right = Point(top_left.x + w, top_left.y + h);
        rectangle(frame, top_left, bottom_right, color, 2);

        Point frame_center = Point(frame.cols / 2, frame.rows / 2);
        double angle = calculate_angle(top_left, frame_center);
        printf("%s arrow detected at angle: %.2f\n", label, angle);
    }
}

VideoCapture init_video_capture() {
    log_initialization();
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        printf("Cannot open webcam\n");
    }
    return cap;
}
