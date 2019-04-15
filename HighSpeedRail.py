from selenium import webdriver

from selenium.webdriver.support.select import Select
from PIL import Image
from time import sleep
import pytesseract
import numpy as np
import cv2
import subprocess
import matplotlib.pyplot as plt

driver = webdriver.Chrome()
driver.get("https://irs.thsrc.com.tw/IMINT/")
driver.maximize_window()

driver.find_element_by_id("btn-confirm").click()
sleep(0.5)

def station():
    selectStartStation = Select(driver.find_element_by_name("selectStartStation"))  # 啟程站
    selectStartStation.select_by_visible_text(u"台北")
    # selectStartStation.select_by_index(2)
    # selectStartStation.select_by_value("2")

    selectStartStation = Select(driver.find_element_by_name("selectDestinationStation"))  # 到達站
    selectStartStation.select_by_visible_text(u"桃園")

def trainCon():
    driver.find_element_by_id("trainCon:trainRadioGroup_0").click()  # 標準車廂
    # driver.find_element_by_id("trainCon:trainRadioGroup_1").click()     #商務車廂

def seatRadio():
    # driver.find_element_by_id("seatRadio0").click()     #無
    driver.find_element_by_id("seatRadio1").click()  # 靠窗優先
    # driver.find_element_by_id("seatRadio2").click()     #走道優先


driver.find_element_by_id("bookingMethod_0").click()        #依照時間搜尋合適車次
# driver.find_element_by_id("bookingMethod_1").click()      #直接輸入車次號碼

def time():
    driver.find_element_by_id("toTimeInputField").clear()
    driver.find_element_by_id("toTimeInputField").send_keys("2019/04/15")  # 去程日期

    toTimeTable = Select(driver.find_element_by_name("toTimeTable"))  # 去程時間
    toTimeTable.select_by_visible_text(u"09:00")  # 00:00~23:30  每隔半小時

def ticket():
    ticketPanel = Select(driver.find_element_by_name("ticketPanel:rows:0:ticketAmount"))  # 全票
    ticketPanel.select_by_visible_text(u"1")  # 全票張數 0~10張

    # toTimeTable = Select(driver.find_element_by_name("ticketPanel:rows:1:ticketAmount"))        #孩童票(6-11歲)
    # toTimeTable.select_by_visible_text(u"2")
    #
    # toTimeTable = Select(driver.find_element_by_name("ticketPanel:rows:2:ticketAmount"))        #愛心票
    # toTimeTable.select_by_visible_text(u"2")
    #
    # toTimeTable = Select(driver.find_element_by_name("ticketPanel:rows:3:ticketAmount"))        #敬老票(65歲以上)
    # toTimeTable.select_by_visible_text(u"2")

station()
trainCon()
seatRadio()
time()
ticket()

def passCode():
    driver.save_screenshot("./img_screenshot.png")
    element = driver.find_element_by_id('BookingS1Form_homeCaptcha_passCode')

    left = element.location['x']
    top = element.location['y']
    right = element.location['x'] + element.size['width']
    bottom = element.location['y'] + element.size['height']

    img = Image.open("./img_screenshot.png")
    img2 = img.crop((left, top, right, bottom))
    img2.save("./passCode.png")

    img = cv2.imread("./passCode.png")
    dst = cv2.fastNlMeansDenoisingColored(img, None, 30, 30, 7, 21)

    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(dst)
    plt.show()

    ret, thresh = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY_INV)
    plt.imshow(thresh)
    plt.show()

    imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    imgarr[:, 3:135] = 0

    import numpy as np
    from sklearn.preprocessing import binarize
    imagedata = np.where(imgarr == 255)

    import matplotlib.pyplot as plt
    plt.scatter(imagedata[1], 47 - imagedata[0], s=100, c='red', label='Cluster 1')
    plt.ylim(ymin=0)
    plt.ylim(ymax=47)
    plt.show()

    X = np.array([imagedata[1]])
    Y = 47 - imagedata[0]

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression

    poly_reg = PolynomialFeatures(degree=2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)

    X2 = np.array([[i for i in range(0, 119)]])

    X2_ = poly_reg.fit_transform(X2.T)

    plt.scatter(X, Y, color="black")
    plt.ylim(ymin=0)
    plt.ylim(ymax=47)
    plt.plot(X2.T, regr.predict(X2_), color="blue", linewidth=3)

    newimg = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    for ele in np.column_stack([regr.predict(X2_).round(0), X2[0], ]):
        pos = 47 - int(ele[0])
        # if newimg[pos-4:pos+4,int(ele[1])] == 255:
        # newimg[pos-3:pos+3,int(ele[1])] = 0
        newimg[pos - 3:pos + 3, int(ele[1])] = 255 - newimg[pos - 3:pos + 3, int(ele[1])]

    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(thresh)
    plt.subplot(122)
    plt.imshow(newimg)
    plt.axis('off')
    plt.imsave("./newimg.png", newimg)
    plt.show()

    # for col in range(3):
    #     count = 0
    #     for i in range(len(imageINV)):
    #         for j in range(len(imageINV[i])):
    #             if (imageINV[i, j, col] == 255):
    #                 count = 0
    #                 for k in range(-2, 3):
    #                     # print(k)
    #                     for l in range(-2, 3):
    #                         try:
    #                             if imageINV[i + k, j + l, col] == 255:
    #                                 count += 1
    #                         except IndexError:
    #                             pass
    #                     # 這裡 threshold 設 4，當周遭小於 4 個點的話視為雜點
    #             if count <= 4:
    #                 imageINV[i, j, col] = 0

    # dilation = cv2.dilate(imageINV, (2,2), iterations=1)
    # cv2.imwrite("./dilation.png", dilation)


    pytesseract.pytesseract.tesseract_cmd = "D://Tesseract-OCR/tesseract.exe"
    image = Image.open("./newimg.png")
    code = pytesseract.image_to_string(image)
    print(code)

passCode()



