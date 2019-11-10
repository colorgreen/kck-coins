import cv2
import glob
import tpls
import numpy as np

templates = tpls.Templates()
templates.load()

def process(image):
    print(image)
    image = cv2.resize(image, ( int(image.shape[1]*700/image.shape[0]), 700 ))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    
    gray_blur = cv2.GaussianBlur(gray, (25, 25), 0)
    thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 1)
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE,
    kernel, iterations=4)
    cont_img = closing.copy()
    contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 7000 or area > 17000:
            continue
        if len(cnt) < 5:
            continue
        ellipse = cv2.fitEllipse(cnt)
        cv2.ellipse(image, ellipse, (0,255,0), 2)
        
    cv2.imshow("Morphological Closing", closing)
    cv2.imshow("Adaptive Thresholding", thresh)
    cv2.imshow('Contours', image)
    cv2.waitKey(0)

    return image

def detect(image):
    for tpl in templates.templates:

        d = cv2.matchShapes(image,tpl.image,cv2.CONTOURS_MATCH_I2,0)

        if d < 0.0001:
            print("Coin detected value: ",tpl.value)
            continue

    return image


if __name__ == "__main__":

    dir = 'data-easy'
    # dir = 'templates-rotated'

    for file in glob.glob(dir+"/*.*"):
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        image = process(image)

        # image = detect(image)

        cv2.imshow('image',image)
        cv2.waitKey(0)
