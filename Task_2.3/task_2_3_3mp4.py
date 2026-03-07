import cv2 as cv
import numpy as np

scale = 1
delta = 0
ddepth = cv.CV_16S

vid = cv.VideoCapture("3.mp4")
bgsub = cv.createBackgroundSubtractorMOG2(detectShadows=False)

while True:
    success, frame = vid.read()
    if not success or frame is None:
        break

    subfr = bgsub.apply(frame)
    medianbl = cv.medianBlur(subfr, 5)
    gauss = cv.GaussianBlur(medianbl, (3, 3), 0.5)
    close = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    morph = cv.morphologyEx(gauss, cv.MORPH_CLOSE, close)

    sx = cv.Sobel(morph, cv.CV_64F, 1, 0, ksize=3)
    sy = cv.Sobel(morph, cv.CV_64F, 0, 1, ksize=3)

    totder = np.sqrt(sx**2 + sy**2)
    binary = totder > 50
    binary = binary.astype(np.uint8) #converting to 0s and 1s 

    height, width = binary.shape
    diag = int(np.sqrt(height**2 + width**2))

    rhos = np.arange(-diag, diag + 1, 1)
    thetas = np.deg2rad(np.arange(0, 180))

    vote = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    non_zero_y, non_zero_x = np.nonzero(binary)

    for i in range(len(non_zero_x)):
        x = non_zero_x[i]
        y = non_zero_y[i]
        for theta_index in range(len(thetas)):
            theta = thetas[theta_index]
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_index = rho + diag
            vote[rho_index, theta_index] += 1


    index = np.argsort(vote.flatten())[::-1]

    
    rho_index1, theta_index1 = np.unravel_index(index[0], vote.shape)

    
    for i in index[1:]:
        rho_index2, theta_index2 = np.unravel_index(i, vote.shape)
        if abs(thetas[theta_index1] - thetas[theta_index2]) < np.deg2rad(5):
            break

    rho1 = rhos[rho_index1]
    theta1 = thetas[theta_index1]

    rho2 = rhos[rho_index2]
    theta2 = thetas[theta_index2]
    

    medianrho = (rho1 + rho2) / 2
    mediantheta = (theta1 + theta2) / 2

    a = np.cos(mediantheta)
    b = np.sin(mediantheta)

    x0 = a * medianrho
    y0 = b * medianrho

    dx = -b
    dy = a

    x1 = int(x0 + 1000 * dx)
    y1 = int(y0 + 1000 * dy)

    x2 = int(x0 - 1000 * dx)
    y2 = int(y0 - 1000 * dy)

    cv.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
    cv.imshow("result", frame)

    if cv.waitKey(30) & 0xFF == 27:
        break

vid.release()
cv.destroyAllWindows()