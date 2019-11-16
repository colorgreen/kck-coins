import cv2
import glob
import tpls
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


def splitCoins(image):
    global templates

    image = cv2.resize(image, (int(image.shape[1]*700/image.shape[0]), 700))
    closing,thresh = tpls.Templates.processToMark(image)

    cont_img = closing.copy()
    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)

    copyImage = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 7000 or area > 25000:
            continue
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)

        coin = getCoin(image, ellipse)
        value = templates.detectCoinValue(coin)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        x,y,w,h = cv2.boundingRect(cnt)

        cv2.ellipse(copyImage, ellipse, (0, 255, 0), 2)

        if value is not None:
            cv2.putText(copyImage, str(value), (x,y), font, 1, (0, 0, 255), 2, cv2.LINE_AA)


    # cv2.imshow("Morphological Closing", closing)
    # cv2.imshow("Adaptive Thresholding", thresh)
    #cv2.waitKey(0)

    return copyImage


if __name__ == "__main__":

    dir = 'data-easy'
    #dir = 'data-medium'
    # dir = 'templates-rotated'

    for file in glob.glob(dir+"/c1.jpg"):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = splitCoins(image)

        cv2.imshow('image', image)
        cv2.waitKey(0)
