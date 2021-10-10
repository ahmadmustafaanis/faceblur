import cv2
import sys

sys.path.append("/home/ahmad/Desktop/faceblur/f3DDFA_V2")
from f3DDFA_V2.FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX

face_boxes = FaceBoxes_ONNX()

inp_path = "video.mp4"
cap = cv2.VideoCapture(inp_path)

output_path = "output.avi"
frame_width = int(cap.get(3))

frame_height = int(cap.get(4))

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc("M", "J", "P", "G"),
    45,
    (frame_width, frame_height),
)

if cap.isOpened() is False:

    print("Error opening video stream or file")

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        face = face_boxes(frame)
        if len(face):
            x1, y1, x2, y2, _ = face[0]
            roi = frame[int(y1) : int(y2), int(x1) : int(x2)]

            cutFrame = cv2.medianBlur(roi, 9)
            cutFrame = cv2.medianBlur(cutFrame, 9)
            cutFrame = cv2.medianBlur(cutFrame, 9)

            frame[int(y1) : int(y2), int(x1) : int(x2)] = cutFrame
            frame = cv2.rectangle(
                frame,
                (int(face[0][0]), int(face[0][1])),
                (int(face[0][2]), int(face[0][3])),
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("frame", frame)
            out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    else:
        print("HERE")

cv2.destroyAllWindows()
