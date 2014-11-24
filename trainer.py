# coding: utf-8

__author__ = 'rchibana'

from SimpleCV import *


class Trainer():

    def __init__(self):
        self.images = None
        self.labels = None
        self.cam = Camera()
        self.outfile = 'test.csv'

    def do_the_train(self):
        f = FaceRecognizer()
        print f.train(self.images, self.labels)
        f.save(self.outfile)
        return f

