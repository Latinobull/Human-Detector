import cv2
import numpy as np

# Main Set up
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()
caps = cv2.VideoCapture(0)

out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 15.0, (640, 480))

while True:
    ret, frame = caps.read()
    person = 1
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in boxes])

    for (xA, yA, xB, yB) in boxes:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"person {person}",
            (xA, yA),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
        person += 1
    cv2.putText(
        frame,
        "Staus: Searching for People",
        (40, 40),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Total People Detected : {person-1}",
        (40, 70),
        cv2.FONT_HERSHEY_DUPLEX,
        0.8,
        (255, 0, 0),
        2,
    )

    out.write(frame.astype("uint8"))
    cv2.imshow("Testing Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

caps.release()
out.release()
cv2.destroyAllWindows()
cv2.waitKey(1)