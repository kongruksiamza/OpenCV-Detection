import cv2

def draw_boundary(img,classifier,scaleFactor,minNighbors,color,text):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features=classifier.detectMultiScale(gray,scaleFactor,minNighbors)
        coords=[]
        for(x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,1,cv2.LINE_AA)
                coords=[x,y,w,h]
                return coords,img
        
def detect(img,faceCascade):
        color = {"blue":(255,0,0), "red":(0,0,255), "green":(0,255,0), "white":(255,255,255)}
        coords = draw_boundary(img, faceCascade, 1.1, 10, color['blue'], "Face")
        return img
        
cap = cv2.VideoCapture(0)
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while (True):
        ret,frame = cap.read()
        frame = detect(frame, faceCascade)
        cv2.imshow("face detection", frame)
        if(cv2.waitKey(1) & 0xFF== ord('q')):
            break
cap.release()
cv2.destroyAllWindows()

        


