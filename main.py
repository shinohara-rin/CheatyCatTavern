from PIL import ImageGrab
import pyautogui
import numpy as np
import cv2 as cv
import win32gui
from time import sleep
from crnn import CRNN
from ppocr import PPOCRDet

det = PPOCRDet('./text_detection_en_ppocrv3_2023may_int8.onnx')
model = CRNN('./text_recognition_CRNN_EN_2022oct_int8.onnx')

def enum_windows_func(hwnd, windows):
    windows.append((hwnd, win32gui.GetWindowText(hwnd)))

windows = []
win32gui.EnumWindows(enum_windows_func, windows)

gamewindow = next((x[0] for x in windows if 'Happy Cat Tavern' in x[1]), None)

while True:
    while not win32gui.GetForegroundWindow() == gamewindow:
        # print('Game window is not at foreground')
        sleep(0.02)

    img = ImageGrab.grab()
    img = np.array(img)

    # img = cv.rectangle(img, ((1920//2)-400,200), ((1920//2)+400, 300), (255, 0, 0), 2)
    img = img[(0):(300), (img.shape[1]//2)-400:(img.shape[1]//2)+400]

    tresh = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, tresh = cv.threshold(tresh, 253, 255, cv.THRESH_BINARY)
    avg = np.average(tresh)
    if avg < 1:
        print('not enough contrast')
        continue

    img = cv.resize(img,(736,736))

    bb = det.infer(img)

    if len(bb[0]) == 0: 
        print('no text')
        continue

    max = np.argmax(bb[1])

    if bb[1][max] < 0.8: 
        print('low confidence')
        continue


    text = model.infer(img, bb[0][max].reshape(8))

    print(text)
    pyautogui.write(text, interval=0.01)
    # cv.imshow('image', cv.cvtColor(img, cv.COLOR_RGB2BGR))
    # cv.waitKey(0)
    # cv.destroyAllWindows()

