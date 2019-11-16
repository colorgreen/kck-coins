import cv2
import glob
import os
import numpy as np

class Templates:

    COIN_UNKNOWN = -1
    COIN_CUPRUM = 1
    COIN_SILVER = 2
    COIN500 = 500
    COIN200 = 200

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


    def createCircleMask(image, lr, sr):

        mask = np.zeros(image.shape[:2], np.uint8)
        w = image.shape[0]
        h = image.shape[1]
        mask = cv2.ellipse(mask, ((w/2, h/2), (w*lr, h*lr), 360), (255, 255, 255), -1)
        mask = cv2.ellipse(mask, ((w/2, h/2), (w*sr, h*sr), 360), (0,0,0), -1)

        return mask

    def saturate(coin, v = 1.8):
        hls = cv2.cvtColor( coin, cv2.COLOR_RGB2HLS )
        hls[:,:,2] =  hls[:,:,2] * v
        return cv2.cvtColor( hls, cv2.COLOR_HLS2RGB )

    def hsvToGlobalRanges( hsv ):
        return ( np.interp( hsv[0], [0,179], [0,360] ), hsv[1]/255, hsv[2]/255 )

    # return single color (b,g,r) -> (h,s,v)
    def colorToHSV( color ):
        hsv = np.uint8([[[color[0],color[1],color[2]]]]) 
        hsvGlobal = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)[0,0]
        return Templates.hsvToGlobalRanges( hsvGlobal )

    # saturated coin array, mask, returns hue
    def extractHSV( saturated, mask ):
        meanColor = cv2.mean(saturated, mask)
        return Templates.colorToHSV(meanColor)

    # saturated coin array, returns (hue, saturation, value)
    def detectColorInLargeCircle(saturated):
        mask = Templates.createCircleMask(saturated, 0.9, 0.8)
        return Templates.extractHSV(saturated, mask) 

    # saturated coin array, returns (hue, saturation, value)
    def detectColorInSmallCircle(saturated):
        mask = Templates.createCircleMask(saturated, 0.5, 0.4)
        return Templates.extractHSV(saturated, mask) 

    def isSilver( hsv ):
        (h, s, v) = hsv

        if 16 <= h <= 27 and s <= 0.40 and s > v:
            return True
        if 14 <= h <= 35 and s <= 0.30:
            return True
        if 9 <= h <= 35 and s + v <= 1.3 and s > v:
            return True

        return False

    def isGold( hsv ):
        (h, s, v) = hsv

        if 32 <= h <= 43 and s >= 0.55:
            return True

        if 28 <= h <= 47 and s >= 0.55:
            return True

        return False

    def isCuprum( hsv ):
        (h, s, v) = hsv

        if 32 <= h <= 43 and s >= 0.65:
            return True

        if 28 <= h <= 47 and s >= 0.55:
            return True

        return False

    def checkCoinType(self, coin):

        hsvL = Templates.detectColorInLargeCircle(coin)
        hsvS = Templates.detectColorInSmallCircle(coin)

        print( hsvL, Templates.isGold(hsvL), Templates.isSilver(hsvL), Templates.isCuprum(hsvL) )
        print( hsvS, Templates.isGold(hsvS), Templates.isSilver(hsvS), Templates.isCuprum(hsvS) )

        if Templates.isGold(hsvL) and Templates.isSilver(hsvS):
            return Templates.COIN200

        if Templates.isSilver(hsvL) and Templates.isGold(hsvS):
            return Templates.COIN500

        if Templates.isSilver(hsvL) and Templates.isSilver(hsvS):
            return Templates.COIN_SILVER

        if Templates.isCuprum(hsvL) and Templates.isCuprum(hsvS):
            return Templates.COIN_CUPRUM

        return Templates.COIN_UNKNOWN

    # coin - rgb array
    def detectCoinValue(self, coin):
        closing,thresh = Templates.processToMark(coin)

        self.i += 1
        # cv2.imwrite("cutted/"+str(self.i)+".png", coin )

        saturated = Templates.saturate(coin)

        coinType = self.checkCoinType(saturated)
        if coinType == Templates.COIN_UNKNOWN:
            coinType = self.checkCoinType(coin)

        cv2.imshow("saturated coin", saturated )
        cv2.imshow("coin", coin)

        print("Coin type: ", coinType)
        cv2.waitKey(0)
        return coinType
        if coinType == Templates.COIN500:
            return 500
        elif coinType == Templates.COIN200:
            return 200
        elif coinType == Templates.COIN_SILVER:
            return self.detectByHuMoments(coin, [10,20,50])
        elif coinType == Templates.COIN_CUPRUM:
            return self.detectByHuMoments(coin, [1,2,5])

        return self.detectByHuMoments(coin)

    def detectByHuMoments(self, coin, potential = [1,2,5,10,20,50,200,500] ):
        deduction = 1000
        value = None
        
        for tpl in self.templates:
            if not tpl.value in potential:
                continue

            # tpl.image - rgb image array with values from templates/
            tplImage = cv2.cvtColor(tpl.image, cv2.COLOR_RGB2GRAY)
            d = cv2.matchShapes(cv2.cvtColor(coin, cv2.COLOR_RGB2GRAY), tplImage, cv2.CONTOURS_MATCH_I2, 0)

            tplClosing,tplThresh = Templates.processToMark(tpl.image)

            # tplThresh = cv2.Canny(tplThresh,200,800)
            
            print(d, tpl.value)
            if d < deduction:
                value = tpl.value
                deduction = d
        
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
