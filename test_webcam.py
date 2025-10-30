import cv2

# Open the default webcam (0 = first camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("✅ Webcam opened successfully. Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame in a window
    cv2.imshow("Webcam Test", frame)

    # If you press the 'q' key, the loop will stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close window
cap.release()
cv2.destroyAllWindows()
