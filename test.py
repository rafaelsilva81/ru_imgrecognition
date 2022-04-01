# Import required packages
import cv2
import pytesseract
import imutils
import utils

# Mention the installed location of Tesseract-OCR in your system
pytesseract.pytesseract.tesseract_cmd = 'C:/cygwin64/bin/tesseract.exe'
 
# Read image from which text needs to be extracted
img = cv2.imread("cardapio-almoco.png")
 
# Preprocessing the image starts
 
# Convert the image to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# Performing OTSU threshold
ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
utils.showImg(thresh1)

# Specify structure shape and kernel size.
# Kernel size increases or decreases the area
# of the rectangle to be detected.
# A smaller value like (10, 10) will detect
# each word instead of a sentence.
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
 
# Applying dilation on the threshold image
dilation = cv2.dilate(thresh1, rect_kernel, iterations = 1)
utils.showImg(dilation)

# Finding contours
contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
                                                 


# A text file is created and flushed
file = open("recognized.txt", "w+", encoding="utf-8")
file.write("")

text = pytesseract.image_to_string(thresh1, lang="por", config='--psm 1')
print(text)
file.write(text)
file.close()
cv2.imwrite('./results/test.png', thresh1)
