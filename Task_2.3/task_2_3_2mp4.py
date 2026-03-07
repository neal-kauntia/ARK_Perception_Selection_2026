import cv2 as cv
import numpy as np

scale = 1
delta = 0
ddepth = cv.CV_16S

vid = cv.VideoCapture("2.mp4")
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows=False)

while True:
    success, frame = vid.read()
    if not success or frame is None:
        break

    fgmask = fgbg.apply(frame)
    medianbl = cv.medianBlur(fgmask, 5)
    gauss = cv.GaussianBlur(medianbl, (3, 3), 0.5)
    kernel_close = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
    morph = cv.morphologyEx(gauss, cv.MORPH_CLOSE, kernel_close)

    sobelx = cv.Sobel(morph, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(morph, cv.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(sobelx**2 + sobely**2)
    edges = magnitude > 50
    edges = edges.astype(np.uint8)

    height, width = edges.shape
    diag_len = int(np.sqrt(height**2 + width**2))

    rhos = np.arange(-diag_len, diag_len + 1, 1)
    thetas = np.deg2rad(np.arange(0, 180))

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(edges)

    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for theta_idx in range(len(thetas)):
            theta = thetas[theta_idx]
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_index = rho + diag_len
            accumulator[rho_index, theta_idx] += 1


    # Sort accumulator values
    indices = np.argsort(accumulator.flatten())[::-1]

    # First strongest line
    rho_idx1, theta_idx1 = np.unravel_index(indices[0], accumulator.shape)

    # Find second strongest line that is parallel
    for idx in indices[1:]:
        rho_idx2, theta_idx2 = np.unravel_index(idx, accumulator.shape)
        #To check for parallel edge lines 
        if abs(thetas[theta_idx1] - thetas[theta_idx2]) < np.deg2rad(5):
            break

    # Convert to actual values
    rho1 = rhos[rho_idx1]
    theta1 = thetas[theta_idx1]

    rho2 = rhos[rho_idx2]
    theta2 = thetas[theta_idx2]
    # Stage 5: Medial axis
    

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

    cv.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv.imshow("result", frame)

    if cv.waitKey(30) & 0xFF == 27:
        break

vid.release()
cv.destroyAllWindows()