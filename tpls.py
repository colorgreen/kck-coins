import cv2
import glob
import os


class Templates:

    def load(self):
        print("Loading templates")

        self.templates = []
        dir = 'templates'

        for file in glob.glob(dir+"/*.png"):
            t = Template(file)
            self.templates.append(t)

            print("Loaded "+file)


class Template:

    def __init__(self, path):

        self.path = path
        self.value = int(os.path.splitext(os.path.basename(path))[0])
        self.image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        moments = cv2.moments(self.image)
        self.huMoments = cv2.HuMoments(moments)
