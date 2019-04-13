import cv2
img=cv2.imread('elephant.jpg')
img=cv2.putText(img,"OpenCV",(10,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
cv2.imshow("Result",img)

cv2.waitKey(0)
cv2.destroyAllWindows()
