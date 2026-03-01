import cv2 as cv

img = cv.imread("noisy.jpg")
cv.imshow("img",img)
gaus= cv.GaussianBlur(img,(3,3),0.5)
cv.imshow("gaus",gaus)
result = cv.fastNlMeansDenoisingColored(gaus, None,h=13,hColor= 13, templateWindowSize=3,searchWindowSize= 8)



cv.imwrite("scenary_final.png",result)
cv.imshow("final",result)


cv.waitKey(0)
cv.destroyAllWindows()
