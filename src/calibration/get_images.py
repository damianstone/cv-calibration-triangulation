import cv2

cap = cv2.VideoCapture(0)
cap2 = cv2.VideoCapture(2)

num = 0

while cap.isOpened():

    succes1, img1 = cap.read()
    success2, img2 = cap2.read()

    k = cv2.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.imwrite('images/stereoLeft/imageL' + str(num) + '.png', img1)
        cv2.imwrite('images/stereoRight/imageR' + str(num) + '.png', img2)
        print("image saved!")
        num += 1

    cv2.imshow('Img1',img1)
    cv2.imshow('Img2',img2)

# Release and destroy all windows before termination
cap.release()

cv2.destroyAllWindows()