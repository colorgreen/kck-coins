import cv2
import glob
import tpls
from coins import Coins
import numpy as np

templates = tpls.Templates()
templates.load()


def crop_image(image, threshold=0):
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    rows = np.where(np.max(flatImage, 0) > threshold)[0]
    if rows.size:
        cols = np.where(np.max(flatImage, 1) > threshold)[0]
        image = image[cols[0]: cols[-1] + 1, rows[0]: rows[-1] + 1]
    else:
        image = image[:1, :1]

    return image


def getCoin(image, ellipse):
    mask = np.zeros_like(image)
    mask = cv2.ellipse(mask, ellipse, color=(255, 255, 255), thickness=-1)
    result = np.bitwise_and(image, mask)

    return crop_image(result, 0)

def getCoinCircle(image, center, r):
    mask = np.zeros_like(image)
    mask = cv2.circle(mask, center, r, color=(255, 255, 255), thickness=-1)
    result = np.bitwise_and(image, mask)

    return crop_image(result, 0)

# list of coins [x,y,r], average r, will delete coins += 4% error margin
def clearCoinsByRadius( coins, r, percent=5 ):
    l = []
    for coin in coins:
        if r*(1-percent/100) <= coin[2] <= r*(1+percent/100):
            l.append(coin)

    return np.array(l)

# from net
def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num 

def getMostFrequentCoinValue( image, coins ):
    l = []
    for (x,y,r) in coins:
        coin = getCoinCircle(image, (int(x),int(y)), r)

        value = templates.detectCoinValue(coin)
        l.append(value)

    return most_frequent(l)

def splitCoins(image):
    global templates

    image = cv2.resize(image, (int(image.shape[1]*900/image.shape[0]), 900))
    copyImage = image.copy()
    silverCoins = []
    cuprumCoins = []
    fiveCoins = []
    twoCoins = []
    allCoins = []
    unknownCoins = []

    circles = cv2.HoughCircles(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT,
            1, 50, param1=130, param2=60, minRadius=20, maxRadius=110)
    if circles is not None:
        circlesSorted = np.sort(circles[0].view('f4,f4,f4'), order=['f2'], axis=0).view(np.float32)[::-1]

        maxR = circlesSorted[0][2]
        biggestCoins = clearCoinsByRadius(circlesSorted, maxR, 10)
        
        avgBiggestSize = Coins.getAverageCoinSizeFromArray(biggestCoins)

        biggestCoinsValue = getMostFrequentCoinValue(image, biggestCoins)
        print("biggestCoinsValue", biggestCoinsValue)
        print(biggestCoins)
        Coins.coins[str(biggestCoinsValue)] = biggestCoins
        for x, y, r in circlesSorted:

            coin = getCoinCircle(image, (int(x),int(y)), r)
            value = templates.detectCoinValue(coin)
            
            cv2.circle(copyImage, (int(x),int(y)), r, (0, 0, 255), 1)
            
            if value != templates.COIN_UNKNOWN:
                allCoins.append([x,y,r])

            if value == templates.COIN500:
                fiveCoins.append( [x,y,r] )
            elif value == templates.COIN200:
                twoCoins.append( [x,y,r] )
            elif value == templates.COIN_CUPRUM:
                cuprumCoins.append( [x,y,r] )
                continue
            elif value == templates.COIN_SILVER:
                silverCoins.append( [x,y,r] )
                continue
            elif value == templates.COIN_UNKNOWN:
                unknownCoins.append( [x,y,r] )
                continue

            cv2.circle(copyImage, (int(x),int(y)), r, (0, 0, 255), 1)

        fiveCoins = np.array(fiveCoins)
        twoCoins = np.array(twoCoins)
        cuprumCoins = np.array(cuprumCoins)
        silverCoins = np.array(silverCoins)
        unknownCoins = np.array(unknownCoins)
        
        for coinVal, coinSize in Coins.sizes.items():
            if coinVal == biggestCoinsValue:
                continue
            print(coinVal, " ps: ", avgBiggestSize*coinSize/Coins.sizes[str(biggestCoinsValue)])
            
            coins = []

            if int(coinVal) in [1,2,5]:
                coins = cuprumCoins
            elif int(coinVal) in [10,20,50,100]:
                coins = silverCoins
            elif int(coinVal) in [200]:
                coins = twoCoins
            elif int(coinVal) in [500]:
                coins = fiveCoins

            # if coinVal == "500":
                # print(Coins.coins["500"])

            Coins.coins[coinVal] = clearCoinsByRadius(coins, 
                    avgBiggestSize*coinSize/Coins.sizes[str(biggestCoinsValue)])

        print(unknownCoins)
        for coinVal, coinSize in Coins.sizes.items():
            coins = []
            for coin in Coins.coins[coinVal]:
                for ccv, ccs in Coins.sizes.items():
                    if ccv == coinVal:
                        continue
                if not Coins.intersectsCoins( coin, Coins.coins[ccv] ):
                    coins.append(coin)
            Coins.coins[coinVal] = coins


            # cv2.circle(copyImage, (int(255),int(553)), 60, (0, 205, 255), 1)


    # closing,thresh = tpls.Templates.processToMark(image)

    # cont_img = closing.copy()
    # contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
    #                                        cv2.CHAIN_APPROX_SIMPLE)


    # for cnt in contours:
    #     area = cv2.contourArea(cnt)
    #     if area < 7000 or area > 25000:
    #         continue
    #     if len(cnt) < 5:
    #         continue
    #     ellipse = cv2.fitEllipse(cnt)

    #     coin = getCoin(image, ellipse)
    #     value = templates.detectCoinValue(coin)
        
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     x,y,w,h = cv2.boundingRect(cnt)

    #     cv2.ellipse(copyImage, ellipse, (0, 255, 0), 2)

    #     if value is not None:
    #         cv2.putText(copyImage, str(value), (x,y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


    # cv2.imshow("Morphological Closing", closing)
    # cv2.imshow("Adaptive Thresholding", thresh)
    #cv2.waitKey(0)

    return copyImage


if __name__ == "__main__":

    dir = 'data-easy'
    #dir = 'data-medium'
    # dir = 'templates-rotated'

    for file in glob.glob(dir+"/c8.jpg"):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = splitCoins(image)

        image = Coins.draw(image)

        cv2.imshow('image', image)
        cv2.waitKey(0)
