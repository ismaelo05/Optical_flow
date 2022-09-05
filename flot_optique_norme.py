import cv2 as cv
import numpy as np
import os

# The video feed is read in as
# a VideoCapture object



cap = cv.VideoCapture(os.path.join('data', 'video', "/home/ismael/Bureau/M2 IFI SIM/Vision par Ordinateur/tp3/jeu de données/PET.avi"))

ret, frame1 = cap.read()


prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)

hsv[..., 0] = 0
hsv[..., 1] = 0

# i = 0
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('norme.avi', fourcc, 7.0, (978, 499))
print(cap.get(cv.CAP_PROP_FRAME_COUNT))
for frame_id in range(int(cap.get(cv.CAP_PROP_FRAME_COUNT))):
# while (1):
    ret, frame2 = cap.read()
    if ret == False:
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 10, 5, 7, 1.5, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    # print(hsv[...,0])
    print(frame_id)
    w, h = hsv[..., 2].shape
    """for i in range(w):
        for j in range(h):
            if hsv[..., 2].item(i, j) > seuil:
                hsv[..., 2].itemset((i,j), 255)
            else:
                hsv[..., 2].itemset((i, j), 0)
    # print(hsv[..., 2])"""
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    height, width, _ = np.array(frame2).shape
    pourcentage_h = 0.85
    pourcentage_w = 0.65

    final = vis = np.concatenate((cv.resize(frame2, (int(height * pourcentage_h), int(width * pourcentage_w))),
                                  cv.resize(bgr, (int(height * pourcentage_h), int(width * pourcentage_w)))), axis=1)
    cv.imshow('frame2', final)
    # print(final.shape)
    # if i > 570 and i <= 620:
    #    out.write(final)
    # i += 1
    # if i > 620:
    #    break

    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    if cv.waitKey(1) & 0xFF == ord('a'):
        cv.imwrite('opticalfc.png', final)
        # cv.imwrite('opticalhsv.png',bgr)

    prvs = next

# Release handle to the webcam
cap.release()
cv.destroyAllWindows()