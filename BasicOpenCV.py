import cv2
faceCascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eyeCascade=cv2.CascadeClassifier("haarcascade_eye.xml")
noseCascade=cv2.CascadeClassifier("Nariz.xml")
mouthCascade=cv2.CascadeClassifier("Mouth.xml")
smileCascade=cv2.CascadeClassifier("haarcascade_smile.xml")

def draw_boundary(img,classifier,scaleFactor,minNeighbors,color,text):
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        features=classifier.detectMultiScale(gray,scaleFactor,minNeighbors,minSize=(55, 55))
        coords=[]
        for (x,y,w,h) in features:
                cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
                cv2.putText(img,text,(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.8,color,2)
                coords=[x,y,w,h]
        return img,coords 
        
def detect(img,faceCascade,eyeCascade,mouthCascade):
        img,coords=draw_boundary(img,faceCascade,1.1,10,(0,0,255),"Face")
        img,coords=draw_boundary(img,eyeCascade,1.1,10,(255,0,0),"Eye")
        img,coords=draw_boundary(img,mouthCascade,1.1,20,(0,255,0),"Mouth")
        return img

        
cap = cv2.VideoCapture("BNK48.mp4")
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
while (True):
        ret,frame = cap.read()
        frame=detect(frame,faceCascade,eyeCascade,mouthCascade)
        cv2.imshow('frame',frame)
        if(cv2.waitKey(1) & 0xFF== ord('q')):
            break
cap.release()
cv2.destroyAllWindows()


