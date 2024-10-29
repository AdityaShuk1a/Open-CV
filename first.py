import cv2
import numpy as np

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create a black canvas for drawing
canvas = None

# Define lower and upper color range for a finger (tune for your environment)
lower_color = np.array([0, 48, 80])  # HSV lower bound (e.g., for skin tone)
upper_color = np.array([20, 255, 255])  # HSV upper bound

# Previous coordinates for tracking finger movement
prev_x, prev_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip horizontally for natural drawing
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert frame to HSV

    # Create a mask for detecting finger based on color range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize canvas if it's None (same size as frame)
    if canvas is None:
        canvas = np.zeros_like(frame)

    if contours:
        # Find the largest contour by area (assuming it's the finger)
        max_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(max_contour) > 300:  # Ignore small contours
            # Get the center of the contour
            (x, y), radius = cv2.minEnclosingCircle(max_contour)
            x, y = int(x), int(y)

            # Draw a circle at the detected finger position
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # If this is not the first frame, draw a line from the previous point
            if prev_x != 0 and prev_y != 0:
                cv2.line(canvas, (prev_x, prev_y), (x, y), (255, 0, 0), 5)

            # Update previous coordinates
            prev_x, prev_y = x, y
        else:
            # Reset previous coordinates if contour is too small (no finger detected)
            prev_x, prev_y = 0, 0

    # Combine the canvas with the live frame
    combined = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Show the live feed and the drawing canvas
    cv2.imshow("Air Drawing", combined)
    cv2.imshow("Mask", mask)

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
