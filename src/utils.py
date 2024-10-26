import cv2
import numpy as np

def detect_faces_and_bodies(frame, face_cascade, body_cascade):
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Detect upper bodies
    bodies = body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces, bodies

def overlay_clothing(frame, faces, bodies, hat, glasses):
    for (x, y, w, h) in faces:
        # Resize and overlay hat
        resized_hat = cv2.resize(hat, (w, int(h / 2)))  # Resize hat based on face width
        frame = overlay_image(frame, resized_hat, x, y - resized_hat.shape[0], w, h)

        # Resize and overlay glasses
        resized_glasses = cv2.resize(glasses, (w, int(h / 3)))  # Resize glasses based on face width
        frame = overlay_image(frame, resized_glasses, x, y + int(h / 4), w, h)

    return frame

def overlay_image(background, overlay, x, y, width, height):
    # Ensure the overlay stays within frame boundaries
    if y < 0:
        overlay = overlay[-y:, :, :]
        y = 0
    if x < 0:
        overlay = overlay[:, -x:, :]
        x = 0
    if y + overlay.shape[0] > background.shape[0]:
        overlay = overlay[:background.shape[0] - y, :, :]
    if x + overlay.shape[1] > background.shape[1]:
        overlay = overlay[:, :background.shape[1] - x, :]

    # Re-check after cropping whether overlay is still valid
    if overlay.shape[0] == 0 or overlay.shape[1] == 0:
        return background  # Skip overlay if dimensions become invalid

    # Handling alpha channel for transparency
    if overlay.shape[2] == 4:  # If image has alpha channel
        alpha_channel = overlay[:, :, 3] / 255.0  # Normalize alpha channel to [0, 1]
        overlay_rgb = overlay[:, :, :3]  # Extract RGB channels
    else:
        alpha_channel = np.ones(overlay.shape[:2], dtype=overlay.dtype)  # No alpha, full opacity
        overlay_rgb = overlay

    # Define the region of interest (roi) in the background where the overlay will be placed
    roi = background[y:y + overlay.shape[0], x:x + overlay.shape[1]]

    # Blend the overlay with the background
    for c in range(0, 3):  # Iterate over RGB channels
        roi[:, :, c] = (alpha_channel * overlay_rgb[:, :, c] + (1 - alpha_channel) * roi[:, :, c])

    # Place the modified roi back into the background
    background[y:y + overlay.shape[0], x:x + overlay.shape[1]] = roi
    return background
