import cv2 as cv
import numpy as np
scale = 1
delta = 0
ddepth = cv.CV_16S
vid = cv.VideoCapture("1.mp4")
fgbg = cv.createBackgroundSubtractorMOG2(detectShadows = False)
while(True):
    success,frame=vid.read()
    fgmask=fgbg.apply(frame)
    # cv.imshow("mask",fgmask)
    # gray = cv.cvtColor(fgmask, cv.COLOR_BGR2GRAY)
    gauss=cv.GaussianBlur(fgmask,(3,3),0.5)
    medianbl= cv.medianBlur(gauss,7)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    morph = cv.morphologyEx(medianbl,cv.MORPH_OPEN,kernel)
    sobelx = cv.Sobel(morph, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(morph, cv.CV_64F, 0, 1, ksize=3)


    magnitude = np.sqrt(sobelx**2 + sobely**2)
    edges = magnitude > 50   # choose threshold
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
            rho = int(x*np.cos(theta) + y*np.sin(theta))
            rho_index = rho + diag_len
            accumulator[rho_index, theta_idx] += 1
            
    
    threshold = 50
    # lines = []
    rho,theta=0,0
    found=False
    for rho_idx in range(accumulator.shape[0]):
        for theta_idx in range(accumulator.shape[1]):
            if accumulator[rho_idx, theta_idx] > threshold:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]
                found=True
                break
        if found: break


    y1,x1,x2,y2 =  100000000,0,0,0
    # for rho,theta in lines:
    a = np.cos(theta)
    b = np.sin(theta)
    for i in range (edges.shape[0]):  #y
        for j in range(edges.shape[1]):  #x
            if edges[i][j]==1:
                val=rho-(j*a)-(i*b)
                if (val<1) and (val>-1):
                    if i<y1:
                        y1=i
                        x1=j
                    if i>y2:
                        y2=i
                        x2=j
                    

    cv.line(frame, (x1,y1), (x2,y2), (0,255,0), 2)
    cv.imshow("result",frame)
    if cv.waitKey(30) & 0xFF=='d':
        break
vid.release()
cv.destroyAllWindows()