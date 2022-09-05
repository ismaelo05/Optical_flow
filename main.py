# Importation des différentes librairies devant permettre la réalisation de notre projet
import glob
import cv2
import numpy as np
import os
# from matplotlib import pyplot as plt
# import natsort
# import random

cap = cv2.VideoCapture(cv2.samples.findFile("/home/ismael/Bureau/M2 IFI SIM/Vision par Ordinateur/tp3/jeu de données/PET.avi"))
# Reading the first frame
ret, frame1 = cap.read()

prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)

hsv[..., 1] = 255

while (1):
    ret, frame2 = cap.read()
    if ret == False:
        break
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    w, h, e = flow.shape
    for i in range(w):
        for j in range(h):
            print(flow[i,j])
    # print(flow)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    height, width, _ = np.array(frame2).shape
    pourcentage_h = 0.7
    pourcentage_w = 0.4

    final = vis = np.concatenate((cv2.resize(frame2, (int(height * pourcentage_h), int(width * pourcentage_w))),
                                  cv2.resize(bgr, (int(height * pourcentage_h), int(width * pourcentage_w)))), axis=1)

    cv2.imshow('frame2', final)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('a'):
        cv2.imwrite('opticalfc_2.png', final)

    prvs = next
