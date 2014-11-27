__author__ = 'rchibana'

# coding: utf-8

from SimpleCV import *
from trainer import Trainer
import time


def get_face_set(cam, myStr=""):
    # Grab a series of faces and return them.
    # quit when we press escape.
    iset = ImageSet()
    count = 0
    disp = Display((640, 480))
    while disp.isNotDone():
        img = cam.getImage()
        fs = img.findHaarFeatures('face.xml')
        if fs is not None and fs != []:
            fs = fs.sortArea()
            face = fs[-1].crop().resize(100, 100)
            fs[-1].draw()
            iset.append(face)
            count = count + 1
        else:
            img.drawText(u'No have anybody in screen!!', 50, 50, color=Color.RED, fontsize=25)
        img.drawText(myStr, 20, 20, color=Color.GREEN, fontsize=32)
        img.save(disp)
    disp.quit()
    return iset

if __name__ == '__main__':
    # Creating the Camera
    cam = Camera(0)

    # names of people to recognize
    names = ['Rodrigo', 'Leticia']

    # how long to wait between each training session
    waitTime = 10

    # First make sure our camera is all set up.
    get_face_set(cam, "Get Camera Ready! - Press ESC to Exit")
    time.sleep(10)

    labels = []
    imgs = []

    # for each person grab a training set of images
    # and generate a list of labels.
    for name in names:
        myStr = "Training for : " + name
        iset = get_face_set(cam, myStr)
        imgs += iset
        labels += [name for i in range(0,len(iset))]
        time.sleep(waitTime)

    # Create, train, and save the recognizer.
    t = Trainer()
    t.labels = labels
    t.images = imgs
    f = t.do_the_train()

    # show the results
    disp = Display((640,480))
    while disp.isNotDone():
        try:
            img = cam.getImage()
            fs = img.findHaarFeatures('face.xml')
            if fs is not None and fs != []:
                fs = fs.sortArea()
                face = fs[-1].crop().resize(100,100)
                fs[-1].draw()
                name, confidence = f.predict(face)
                img.drawText(name, 30, 30, fontsize=64)
            img.save(disp)
        except Exception, e:
            print e
            continue