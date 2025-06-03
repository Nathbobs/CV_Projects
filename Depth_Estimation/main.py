import cv2
import torch
import matplotlib.pyplot as plt

#donwloading the MiDAS model
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
midas.to('cpu')
midas.eval()

#input transform
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.small_transform

#to openCV
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow("Input", frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        