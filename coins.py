import numpy as np
import cv2
from statistics import median

class Coins:
    coins = {
        "1": [],
        "2": [],
        "5": [],
        "10": [],
        "20": [],
        "50": [],
        "100": [],
        "200": [],
        "500": []
    }

    sizes = {
        "1": 15.50,
        "2": 17.50,
        "5": 19.50,
        "10": 16.50,
        "20": 18.50,
        "50": 20.50,
        "100": 23,
        "200": 21.5,
        "500": 24
    }

    font = cv2.FONT_HERSHEY_SIMPLEX

    # coin a(x,y,r) b(x,y,r)
    def intersects( a, b ):
        return sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2) <= a[2]+b[2]

    def intersectsCoins( a, coins ):
        for coin in coins:
            if Coins.intersects(a, coin):
                return True
        return False

    # array of coins [z,y,r]
    def getAverageCoinSizeFromArray( coins ):
        coins = np.array(coins)

        if coins.size == 0:
            return None
        return (np.mean(coins[:, 2])+np.median(coins[:, 2])) / 2

    def predictFiveSize(coins):
        coins = np.array(coins)
        if coins.size == 0:
            return None
        return (np.mean(coins[:, 2])+np.median(coins[:, 2])) / 2 * 24.0 / 21.50

    # array of (x,y,r)
    def predictTwoSize(coins):
        coins = np.array(coins)
        if coins.size == 0:
            return None
        return (np.mean(coins[:, 2])+np.median(coins[:, 2]))/2 * 21.50 / 24.0

    
    def markCoins( image, coins, value, color= (0, 255, 0) ):

        for coin in coins:
            (x,y,r) = coin
            cv2.circle(image, (int(x),int(y)), r, color, 1)
            cv2.putText(image, str(value), (x,y), Coins.font, 1, color, 2, cv2.LINE_AA) 

    def draw( image ):
        Coins.markCoins(image, Coins.coins["1"], "1")
        Coins.markCoins(image, Coins.coins["2"], "2")
        Coins.markCoins(image, Coins.coins["5"], "5")
        Coins.markCoins(image, Coins.coins["10"], "10")
        Coins.markCoins(image, Coins.coins["20"], "20")
        Coins.markCoins(image, Coins.coins["50"], "50")
        Coins.markCoins(image, Coins.coins["100"], "100")
        Coins.markCoins(image, Coins.coins["200"], "200")
        Coins.markCoins(image, Coins.coins["500"], "500", color=(20,160,240))

        return image
