from flask import Flask, render_template, request, redirect
from cv2 import cv2
import sys
import numpy
import os
import face_recognition
from datetime import datetime, date
import csv
import pandas as pd
from flask_table import Table, Col
import os.path
import imageio
import scipy.ndimage
haar_file = 'haarcascade_frontalface_default.xml'

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        now = str(datetime.now())
        now = now.replace(':', '')
        now = now.replace('-', '')
        now = now.replace('.', '')
        now = now.replace(' ', '')
        if request.files:
            sImage = request.files["img"]
            img = sImage.read()
        else:
            img = openCam()
            now = img
            img = f'webcam\\data\\{now}.png'
            if(img == 0):
                return render_template('home.html')
        imgRead = imageio.imread(img)
        gray = grayscale(imgRead)
        temp = 255-gray

        b = scipy.ndimage.filters.gaussian_filter(temp, sigma=13)
        r = dodge(b, gray)
        cv2.imwrite(f'static\\{now}.png', r)
        print(now)
        now = f'/static/{now}.png'
        print(now)
        return render_template('home.html', user_image=now)
    return render_template('home.html', user_image='')


def openCam():
    datasets = 'webcam'

    sub_data = 'data'

    path = os.path.join(datasets, sub_data)
    if not os.path.isdir(path):
        os.mkdir(path)

    webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    (_, im) = webcam.read()
    now = str(datetime.now())
    now = now.replace(':', '')
    now = now.replace('-', '')
    now = now.replace('.', '')
    now = now.replace(' ', '')
    cv2.imwrite(f'webcam\\data\\{now}.png', im)
    return now


def grayscale(rgb):
    return numpy.dot(rgb[..., :3], [0.2989, 0.5870, .1140])


def dodge(front, back):
    result = front*255/(255-back)
    result[result > 255] = 255
    result[back == 255] = 255
    return result.astype('uint8')


if __name__ == '__main__':
    app.run(debug=True)
