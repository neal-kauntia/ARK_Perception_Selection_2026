import cv2 as cv

img = cv.imread("iron_man_noisy.jpg")
cv.imshow("img",img)

gray= cv.cvtColor(img,cv.COLOR_BGR2GRAY) #Converting to grayscale img
medfil = cv.medianBlur(gray,3)
# cv.imshow("denoised",medfil)

kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,2))
morphopen= cv.morphologyEx(medfil,cv.MORPH_OPEN,kernel)
cv.imwrite("iron_man_final.png",morphopen)
cv.imshow("result",morphopen)
cv.waitKey(0)
cv.destroyAllWindows()
