# Code for real-time webcam capture and preprocessing

import cv2

def start_webcam_feed():
    # Initialize webcam capture
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read each frame from the webcam
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break
        
        # Resize the frame
        resized_frame = cv2.resize(frame, (640, 480))
        
        # Convert to grayscale
        gray_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
        
        # Normalize the pixel values
        normalized_frame = gray_frame / 255.0
        
        # Display the webcam feed
        cv2.imshow("Webcam Feed", resized_frame)
        
        # Break on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the capture and destroy all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    start_webcam_feed()
