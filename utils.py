import cv2

def showImg(img):
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def x_cord_contour(contours):
    #Returns the X cordinate for the contour centroid
    M = cv2.moments(contours)
    return (int(M['m10']/M['m00']))
    
def y_cord_contour(contours):
    #Returns the Y cordinate for the contour centroid
    M = cv2.moments(contours)
    return (int(M['m01']/M['m00']))
