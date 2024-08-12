import cv2 as cv
import numpy as np

def do_canny(frame):
    gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    canny = cv.Canny(blur, 50, 150)
    return canny

def do_segment(frame):
    height = frame.shape[0]
    polygons = np.array([[(0, height), (800, height), (380, 290)]])
    mask = np.zeros_like(frame)
    cv.fillPoly(mask, polygons, 255)
    segment = cv.bitwise_and(frame, mask)
    return segment

def calculate_lines(frame, lines):
    if lines is None:
        return np.array([])  # Return empty array if no lines detected
    
    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        y_intercept = parameters[1]
        if slope < 0:
            left.append((slope, y_intercept))
        else:
            right.append((slope, y_intercept))
    
    left_avg = np.average(left, axis=0) if left else None
    right_avg = np.average(right, axis=0) if right else None
    
    left_line = calculate_coordinates(frame, left_avg) if left_avg is not None else None
    right_line = calculate_coordinates(frame, right_avg) if right_avg is not None else None
    return np.array([left_line, right_line]) if left_line is not None and right_line is not None else np.array([])

def calculate_coordinates(frame, parameters):
    slope, intercept = parameters
    y1 = frame.shape[0]
    y2 = int(y1 - 150)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])

def visualize_lines(frame, lines):
    lines_visualize = np.zeros_like(frame)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv.line(lines_visualize, (x1, y1), (x2, y2), (0, 255, 0), 5)
    return lines_visualize

def is_color_video(frame):
    # Check if the video frame has 3 channels (color)
    return frame.shape[2] == 3

cap = cv.VideoCapture("input.mp4")

if not cap.isOpened():
    print("Error: Video file could not be opened.")
    exit()

# Read the first frame to check if it's a color video
ret, frame = cap.read()
if not ret or not is_color_video(frame):
    print("Error: The video file is not a color video or could not be read.")
    cap.release()
    exit()

# Process the video
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video or error

    canny = do_canny(frame)  # Compute Canny edges but do not display
    segment = do_segment(canny)
    hough = cv.HoughLinesP(segment, 2, np.pi / 180, 100, np.array([]), minLineLength=100, maxLineGap=50)
    lines = calculate_lines(frame, hough)
    lines_visualize = visualize_lines(frame, lines)

    output = cv.addWeighted(frame, 0.9, lines_visualize, 1, 1)
    cv.imshow("output", output)

    # Break the loop if the user presses 'q' or closes the window
    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if cv.getWindowProperty("output", cv.WND_PROP_VISIBLE) < 1:
        break  # Break if the window is closed

cap.release()
cv.destroyAllWindows()
