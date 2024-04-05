
import cv2
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1440)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

while cap.isOpened():
    cap.grab()
    ret, frame = cap.read()
    if ret is True:
        break

cv2.imshow('image', frame)
cv2.waitKey()
