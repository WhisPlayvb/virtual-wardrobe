# Code for detecting body keypoints (shoulders, elbows, etc.)

import cv2

# Initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_body(frame):
    # Detect people in the image
    bodies, _ = hog.detectMultiScale(frame, winStride=(8, 8))
    
    # Draw rectangles around detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return frame, bodies

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect body and display the frame
        frame_with_body, bodies = detect_body(frame)
        cv2.imshow("Body Pose Detection", frame_with_body)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
