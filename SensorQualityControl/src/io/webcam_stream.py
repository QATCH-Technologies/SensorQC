import cv2

# Open the default camera (usually 0 for built-in, 1 for external)
print("[INFO] opening USB camera...")
cap = cv2.VideoCapture(0)

print("[INFO] streaming...")
print("[INFO] press 'q' to stop")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Webcam Viewer', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture
print("[INFO] stopping...")
cap.release()
cv2.destroyAllWindows()