import os
import cv2
from utils import detect_faces_and_bodies, overlay_clothing

def virtual_tryon():
    cap = cv2.VideoCapture(0)

    # Use absolute paths for the clothing and accessories images
    hat_path = r'C:\XDrive\t\virtual-wardrobe2\data\clothing\hat.png'
    glasses_path = r'C:\XDrive\t\virtual-wardrobe2\data\clothing\glasses.png'

    # Load clothing and accessories images
    hat = cv2.imread(hat_path, cv2.IMREAD_UNCHANGED)
    glasses = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)

    # Check if the images are loaded successfully
    if hat is None:
        raise FileNotFoundError(f"Hat image not found at {hat_path}. Please check the path.")
    if glasses is None:
        raise FileNotFoundError(f"Glasses image not found at {glasses_path}. Please check the path.")

    # Load face and body cascades for detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces and bodies in the current frame
        faces, bodies = detect_faces_and_bodies(frame, face_cascade, body_cascade)

        # Overlay clothing and accessories onto the detected keypoints
        frame = overlay_clothing(frame, faces, bodies, hat, glasses)

        # Show the real-time video with virtual try-on
        cv2.imshow('Virtual Try-On', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    virtual_tryon()
