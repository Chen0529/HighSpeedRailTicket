from PIL import Image
Image.open("./passCode.png")

import cv2
img = cv2.imread("./passCode.png")
dst = cv2.fastNlMeansDenoisingColored(img, None, 30, 30 , 7 , 21)

import matplotlib.pyplot as plt
plt.subplot(121)
plt.imshow(img)
plt.subplot(122)
plt.imshow(dst)
plt.show()

ret,thresh = cv2.threshold(dst,127,255,cv2.THRESH_BINARY_INV)
plt.imshow(thresh)
plt.show()

imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
imgarr[:,3:135] = 0

import numpy as np
from sklearn.preprocessing import binarize
imagedata = np.where(imgarr == 255)

import matplotlib.pyplot as plt
plt.scatter(imagedata[1], 47 - imagedata[0], s = 100, c = 'red', label = 'Cluster 1')
plt.ylim(ymin=0)
plt.ylim(ymax=47)
plt.show()

X = np.array([imagedata[1]])
Y = 47 - imagedata[0]

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg= PolynomialFeatures(degree = 2)
X_ = poly_reg.fit_transform(X.T)
regr = LinearRegression()
regr.fit(X_, Y)

X2 = np.array([[i for i in range(0,119)]])

X2_ = poly_reg.fit_transform(X2.T)

plt.scatter(X,Y, color="black")
plt.ylim(ymin=0)
plt.ylim(ymax=47)
plt.plot(X2.T, regr.predict(X2_), color= "blue", linewidth = 3)

newimg =  cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

for ele in np.column_stack([regr.predict(X2_).round(0),X2[0],] ):
    pos = 47-int(ele[0])
    #if newimg[pos-4:pos+4,int(ele[1])] == 255:
    #newimg[pos-3:pos+3,int(ele[1])] = 0
    newimg[pos-3:pos+3,int(ele[1])] = 255 - newimg[pos-3:pos+3,int(ele[1])]

import matplotlib.pyplot as plt
plt.subplot(121)
plt.imshow(thresh)
plt.subplot(122)
plt.imshow(newimg)
plt.axis('off')
plt.imsave("./newimg.png", newimg)
plt.show()

import pytesseract
pytesseract.pytesseract.tesseract_cmd = "D://Tesseract-OCR/tesseract.exe"
image = Image.open("./newimg.png")
code = pytesseract.image_to_string(image)
print(code)

