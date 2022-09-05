# Importation des différents packages
import cv2 as cv
import numpy as np

# Définition du nombre de cluster k
k = 20
# The video feed is read in as
# a VideoCapture object

# Objet contenant la video
cap = cv.VideoCapture("/home/ismael/Bureau/M2 IFI SIM/Vision par Ordinateur/tp3/jeu de données/PET.avi")
# cap = cv.VideoCapture("/home/ismael/Bureau/Rush Hour Traffic with motorcycle in Ho Chi Minh city - Vietnam.mp4")

# Lecture de la vidéo
ret, frame1 = cap.read()

# Lecture du premeir frame
prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)

hsv[..., 0] = 255
hsv[..., 1] = 255
w, h = prvs.shape
# i = 0
# fourcc = cv.VideoWriter_fourcc(*'XVID')
# out = cv.VideoWriter("/home/ismael/Bureau/M2 IFI SIM/Vision par Ordinateur/tp3/norme.avi", fourcc, 7.0, (978, 499))
fourcc = cv.VideoWriter_fourcc(*'XVID')
out = cv.VideoWriter("name_result" + '.avi', fourcc, 9.0, (w * 2, h * 2))
while (1):
    ret, frame2 = cap.read()
    if ret == False:
        break
    next = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 10, 5, 7, 1.5, 0)
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = 255
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # reshape the image to a 2D array of pixels and 3 color values
    pixel_values = hsv[..., 2].reshape((-1, 3))
    # Convertir en réel
    pixel_values = np.float32(pixel_values)
    # definition de la condition d'arrèt
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # Appel de la fonction kMeans
    _, labels, (centers) = cv.kmeans(pixel_values, k, None, criteria, 20, cv.KMEANS_RANDOM_CENTERS)
    # Reconvertion sur 8 bits
    centers = np.uint8(centers)

    # On met le centre max à 255 et le reste à 0
    maximum = centers.max()
    # print()
    for i, v in enumerate(centers):
        # print(i, v)
        # print([maximum])
        if np.array_equal([maximum], v):
            centers[i] = 255
        else:
            centers[i] = 0
    res = centers[labels.flatten()]
    hsv[..., 0] = res.reshape((hsv[..., 2].shape))

    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)


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
    height, width, _ = np.array(frame2).shape
    pourcentage_h = 0.85
    pourcentage_w = 0.65
    final = vis = np.concatenate((cv.resize(frame2, (int(height * pourcentage_h), int(width * pourcentage_w))),
                                  cv.resize(bgr, (int(height * pourcentage_h), int(width * pourcentage_w)))), axis=1)

    cv.imshow('frame2', final)
    # final = cv.flip(final, 0)
    out.write(final)


    # Hit 'q' on the keyboard to quit!
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    prvs = next

# Release handle to the webcam
cap.release()
out.release()
cv.destroyAllWindows()