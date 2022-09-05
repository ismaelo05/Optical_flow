# Importation des différents packages
import cv2 as cv
import numpy as np

# Définition des différents seuillages
seuil1 = 40
seuil2 = 80
seuil3 = 120
seuil4 = 160
seuil5 = 200

# Objet contenant la video
cap = cv.VideoCapture("/home/ismael/Bureau/M2 IFI SIM/Vision par Ordinateur/tp3/jeu de données/PET.avi")
# cap = cv.VideoCapture("/home/ismael/Bureau/Rush Hour Traffic with motorcycle in Ho Chi Minh city - Vietnam.mp4")

# Lecture de la vidéo
ret, frame1 = cap.read()

# Lecture du premeir frame
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

# Mise à zéro de la variable hsv de meme taille que le frame
hsv = np.zeros_like(frame1)



# Mise à 255 les valeurs de hsv
hsv[..., 0] = 255
hsv[..., 1] = 255

# i = 0
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter('norme.avi', fourcc, 7.0, (978, 499))

# Parcours des différents frames de notre vidéo
while (1):
    ret, frame2 = cap.read()
    if ret == False:
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 10, 5, 7, 1.5, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    print(mag)

    hsv[...,0] = ang*180/np.pi/2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    print(hsv[..., 2])
    print(hsv)
    w, h = hsv[..., 2].shape
    # Modification des valeurs de la norme selon la valeur du seuil
    for i in range(w):
        for j in range(h):
            if hsv[..., 2].item(i, j) < seuil1:
                hsv[..., 2].itemset((i,j), 0)
            elif hsv[..., 2].item(i, j) < seuil2:
                hsv[..., 2].itemset((i, j), 50)
            elif hsv[..., 2].item(i, j) < seuil3:
                hsv[..., 2].itemset((i, j), 100)
            elif hsv[..., 2].item(i, j) < seuil4:
                hsv[..., 2].itemset((i, j), 155)
            elif hsv[..., 2].item(i, j) < seuil5:
                hsv[..., 2].itemset((i, j), 205)
            else:
                hsv[..., 2].itemset((i, j), 255)
    print(hsv[..., 2])

    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    cv.imshow('test', bgr)
    cv.waitKey(0)
    break
    """gray = cv.cvtColor(bgr, cv.COLOR_BGR2GRAY)

    # treshold
    _, thresh = cv.threshold(gray, 100, 255, cv.THRESH_BINARY)

    dilated = cv.dilate(thresh, None, iterations=5)

    # define contours
    contours, _ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cnt_area = cv.contourArea(contour)

        if cnt_area < 1000 or cnt_area > 4000:
            continue
        (x, y, w, h) = cv.boundingRect(contour)
        cv.rectangle(bgr, (x, y), (x + w, y + h), (0, 0, 255))
        print(cnt_area)"""
    # Récupération des largeurs et hauteurs des frames
    height, width, _ = np.array(frame2).shape
    pourcentage_h = 0.85
    pourcentage_w = 0.65

    # Concatenation des images
    final = vis = np.concatenate((cv.resize(frame2, (int(height * pourcentage_h), int(width * pourcentage_w))),
                                  cv.resize(bgr, (int(height * pourcentage_h), int(width * pourcentage_w)))), axis=1)

    cv.imshow('frame2', final)
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
# out.release()
cv.destroyAllWindows()