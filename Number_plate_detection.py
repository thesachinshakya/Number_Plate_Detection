import cv2
import numpy as np

frameWidth = 640    #Frame Width
franeHeight = 480   # Frame Height

plateCascade = cv2.CascadeClassifier(r"C:\Users\NAGARAJU\Desktop\sample\Number_Plate_Detection\haarcascade_russian_plate_number.xml")
minArea = 500

cap =cv2.VideoCapture(0)
cap.set(3,frameWidth)
cap.set(4,franeHeight)
cap.set(10,150)
count = 0

while True:
    success , img  = cap.read()

    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade .detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w*h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img,"NumberPlate",(x,y-5),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            imgRoi = img[y:y+h,x:x+w]
            cv2.imshow("ROI",imgRoi)
    cv2.imshow("Result",img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(r"D:\numberplate\images" + str(count) + ".jpg", imgRoi)
        saved_img = img.copy()  # Create a copy of the image for displaying the message
        cv2.rectangle(saved_img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(saved_img, "Scan Saved", (15, 265), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
        cv2.imshow("Result", saved_img)  # Display the image with the message
        cv2.waitKey(500)
        count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


