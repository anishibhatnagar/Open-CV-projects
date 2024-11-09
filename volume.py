import mediapipe as mp
import numpy as np
import time
import cv2
import math
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.25
)
mp_draw = mp.solutions.drawing_utils

pTime = 0
volrange=volume.GetVolumeRange()
minv=volrange[0]
maxv=volrange[1]

while True:
    success, img = cap.read()

    if not success:
        print("Error: Failed to capture image.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lmlist = []  # Initialize the landmark list

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            h, w, _ = img.shape  # Get the dimensions of the image
            for i, lm in enumerate(hand_landmarks.landmark):
                # Convert normalized coordinates to pixel coordinates
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append((i, cx, cy))  # Append landmark index and coordinates

                # Display the index and coordinates of each landmark
                cv2.putText(img, f'{i}', (cx - 20, cy - 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)

    if len(lmlist) > 8:  # Ensure there are at least 9 landmarks
        # Get the coordinates of landmark 4 and 8
        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        length=math.hypot(x2-x1,y2-y1)
        #print(length)
        
        cv2.circle(img, (x1, y1), 10, (255, 255, 0), cv2.FILLED)  # Circle for thumb tip
        cv2.circle(img, (x2, y2), 10, (255, 255, 0), cv2.FILLED)
        # Circle for index finger tip
        cv2.line(img,(x1,y1),(x2,y2),(255,255,0),3)
        vol=np.interp(length,[45,260],[minv,maxv])
        print(vol)
        volume.SetMasterVolumeLevel(vol,None)


    # Calculate and display the frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'{int(fps)}', (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

    # Flip the image horizontally for a mirror effect
    img = cv2.flip(img, 1)
    cv2.imshow('Hand Tracking', img)

    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
