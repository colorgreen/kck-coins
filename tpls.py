import cv2
import glob
import os
import numpy as np

class Templates:

    def __init__(self):
        self.i = 0

    def load(self):
        print("Loading templates")

        self.templates = []
        dir = 'templates'

        for file in glob.glob(dir+"/*.png"):
            t = Template(file)
            self.templates.append(t)

            print("Loaded "+file)

    # coin - rgb array
    def detectCoinValue(self, coin):
        closing,thresh = Templates.processToMark(coin)
        cv2.imshow("coin closing", thresh)

        self.i += 1
        #cv2.imwrite("cutted/"+str(self.i)+".png", coin )

        deduction = 1000
        value = None

        for tpl in self.templates:
            # tpl.image - rgb image array with values from templates/
            tplImage = cv2.cvtColor(tpl.image, cv2.COLOR_RGB2GRAY)
            d = cv2.matchShapes(cv2.cvtColor(coin, cv2.COLOR_RGB2GRAY), tplImage, cv2.CONTOURS_MATCH_I2, 0)

            tplClosing,tplThresh = Templates.processToMark(tpl.image)

            tplThresh = cv2.Canny(tplThresh,200,800)
            cv2.imshow("tpl closing", tplThresh)
            cv2.waitKey
            
            print(d, tpl.value)
            if d < deduction:
                value = tpl.value
                deduction = d
        
        print("Coin detected value: ", tpl.value)
        return value

    
    def processToMark(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray_blur = cv2.GaussianBlur(gray, (25, 25), 0)
        thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 1)
        kernel = np.ones((3, 3), np.uint8)
        return (cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
                               kernel, iterations=4), thresh)



class Template:

    def __init__(self, path):

        self.path = path
        self.value = str(os.path.splitext(os.path.basename(path))[0])
        self.image = cv2.imread(path, cv2.IMREAD_COLOR)

        grey = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        moments = cv2.moments(grey)
        self.huMoments = cv2.HuMoments(moments)
