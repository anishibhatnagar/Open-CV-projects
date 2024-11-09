import cv2
import mediapipe as mp
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

pTime = 0

while True:
    success, img = cap.read()

    if not success:
        print("Error: Failed to capture image.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = img.shape  # Get the dimensions of the image
            for i, lm in enumerate(hand_landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                
                # Draw a purple circle on each landmark
                cv2.circle(img, (cx, cy), 8, (255, 0, 255), cv2.FILLED)
                
                # Display the index and coordinates of each landmark
                cv2.putText(img, f'{i}', (cx - 20, cy - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)

    # Calculate and display the frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'{int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 0, 255), 3)

    # Display the image
    cv2.imshow('Hand Tracking', img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
