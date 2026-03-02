import cv2 as cv

img = cv.imread("noisy.jpg")
cv.imshow("img",img)
gaus= cv.GaussianBlur(img,(3,3),0.5)
# cv.imshow("gaus",gaus)
result = cv.fastNlMeansDenoisingColored(gaus, None,h=15,hColor= 15, templateWindowSize=3,searchWindowSize= 10)
# cv.imshow("result",result)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(2,2))
morph = cv.morphologyEx(result,cv.MORPH_OPEN,kernel)

cv.imwrite("scenary_final.png",morph)
cv.imshow("final",morph)


cv.waitKey(0)
cv.destroyAllWindows()
