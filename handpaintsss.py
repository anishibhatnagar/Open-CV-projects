import numpy as np
import mediapipe as mp
import os
import cv2
import time

# Load header images
folderpath = 'header'
mylist = os.listdir(folderpath)
print(mylist)
overlaylist = []
for impath in mylist:
    image = cv2.imread(f'{folderpath}/{impath}')
    overlaylist.append(image)

header = overlaylist[0]
drawcolor = (255, 255, 0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.85,
    min_tracking_confidence=0.15
)
mp_draw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
xp, yp = 0, 0
imgcanvas = np.zeros((720, 1280, 3), np.uint8)

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # Set width
cap.set(4, 720)   # Set height

while True:
    success, img = cap.read()

    if not success:
        print("Error: Failed to capture image.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    lmlist = []
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
                
                # Append landmark index and coordinates to list
                lmlist.append((i, cx, cy))

    # Process the landmarks if available
    if len(lmlist) >= 21:  # Check if there are at least 21 landmarks
        # Tips of the index and middle fingers
        x1, y1 = lmlist[8][1:]   # Index finger tip
        x2, y2 = lmlist[12][1:]  # Middle finger tip

        # IDs for finger tips
        tipids = [4, 8, 12, 16, 20]
        fingers = []

        # Thumb
        if lmlist[tipids[0]][1] > lmlist[tipids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # Other fingers
        for i in range(1, 5):
            if lmlist[tipids[i]][2] < lmlist[tipids[i] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawcolor, cv2.FILLED)
            print('selection')
            if y1 < 125:
                if 300 < x1 < 500:
                    header = overlaylist[3]
                    drawcolor = (0, 0, 0)
                elif 550 < x1 < 750:
                    header = overlaylist[2]
                    drawcolor = (0, 255, 0)
                elif 800 < x1 < 950:
                    header = overlaylist[1]
                    drawcolor = (255, 0, 0)
                elif 1000 < x1:
                    header = overlaylist[0]
                    drawcolor = (255, 0, 255)
                    
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, (255, 255, 0), cv2.FILLED)
            print('drawing')
            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawcolor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, 85)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), drawcolor, 55)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, 15)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), drawcolor, 15)

            xp, yp = x1, y1

        imggray = cv2.cvtColor(imgcanvas, cv2.COLOR_BGR2GRAY)
        _, imginv = cv2.threshold(imggray, 50, 255, cv2.THRESH_BINARY_INV)
        imginv = cv2.cvtColor(imginv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, imginv)
        img = cv2.bitwise_or(img, imgcanvas)
     
    # Flip the image and canvas horizontally for a mirror effect
    img = cv2.flip(img, 1)

    # Ensure the header size fits the display area
    header_resized = cv2.resize(header, (img.shape[1], header.shape[0]))
    img[0:header_resized.shape[0], 0:header_resized.shape[1]] = header_resized

    # Show the image
    cv2.imshow('Hand Tracking', img)
    #cv2.imshow('Canvas', imgcanvas)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
