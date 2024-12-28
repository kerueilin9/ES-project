#Importing necessary libraries.
import argparse
import time
import RPi.GPIO as GPIO
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, non_max_suppression, apply_classifier, scale_coords, xyxy2xywh, \
    strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

import os
os.environ['QT_QPA_PLATFORM'] = 'xcb'

############################################################ GPIO PIN
# Button GPIO
BUTTON_PIN = 17

# Ultrasound GPIO
TRIGGER_PIN = 22
ECHO_PIN = 27

# Buzzer GPIO
BUZZER_PIN = 10

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
GPIO.setup(TRIGGER_PIN, GPIO.OUT)
GPIO.setup(ECHO_PIN, GPIO.IN)
GPIO.setup(BUZZER_PIN, GPIO.OUT)

############################################################ Buzzer
def beep_buzzer():
    GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(BUZZER_PIN, GPIO.LOW)

############################################################ Ultrasound
def send_trigger_pulse():
    GPIO.output(TRIGGER_PIN, True)
    time.sleep(0.00001)
    GPIO.output(TRIGGER_PIN, False)
    
def wait_for_echo(value, timeout):
    start_time = time.time()
    while GPIO.input(ECHO_PIN) != value:
        if time.time() - start_time > timeout:
            return False
    return True

def get_distance():
    send_trigger_pulse()
    if not wait_for_echo(False, 0.01):
        print("Echo 引腳未準備好")
        return None
    if not wait_for_echo(True, 0.01):
        print("等待高電平超時")
        return None
    start = time.time()
    if not wait_for_echo(False, 0.01):
        print("等待低電平超時")
        return None
    finish = time.time()
    
    pulse_len = finish - start
    distance_cm = (pulse_len * 34300) / 2
    return distance_cm

############################################################ Camera
def capture_image():
    # 捕捉影像
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("無法打開攝像頭")
        exit()
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("image.jpg", frame)
        print("照片已保存為 image.jpg")
        cap.release()

############################################################ 

def detect_single_image(image_path, save_img=False):
    # Source configuration
    source, weights, imgsz = image_path, opt.weights, opt.img_size
    device = select_device(opt.device)
    half = device.type != 'cpu'

    # Load model
    model = attempt_load(weights, map_location=device)
    imgsz = check_img_size(imgsz, s=model.stride.max())
    if half:
        model.half()

    # Set Dataloader
    dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors for bounding boxes
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    for path, img, im0s, _ in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img, augment=opt.augment)[0]
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=3)
                    print(label)

        # Display the image
        # cv2.imshow("Result", im0s)
        if save_img:
            save_path = f"result_{Path(path).name}"
            cv2.imwrite(save_path, im0s)

        # while True:
            # 檢查是否按下按鍵，或是否關閉視窗
            # if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下 'q' 鍵退出
            #     break
            # 檢測視窗是否已關閉
            # if cv2.getWindowProperty("Result", cv2.WND_PROP_VISIBLE) < 1:
            #     break

        # 銷毀所有視窗
        # cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='../Model/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.50, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--classes', default=None, nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    opt = parser.parse_args()
    try:
        with torch.no_grad():
    ###################################################################################            
            print("系統啟動，請按下按鈕拍攝照片...")    
            button_pressed = False  # 用於檢測按鈕是否按下
            while True:
                # 等待按鈕按下
                button_state = GPIO.input(BUTTON_PIN)
                
                if button_state == GPIO.HIGH and not button_pressed:  # 如果按鈕被按下且之前未按下
                    button_pressed = True  # 記錄按鈕已被按下
                    capture_image()
                    time.sleep(1)
                    detect_single_image("image.jpg", True)
                elif button_state == GPIO.LOW and button_pressed:  # 如果按鈕釋放
                    button_pressed = False  # 記錄按鈕已釋放
                
                ultrasound_distance = get_distance()
                if ultrasound_distance is not None:
                    print("cm=%f" % ultrasound_distance)
                    if ultrasound_distance < 5:
                        beep_buzzer()
                time.sleep(1)  # 加入短暫延遲，避免過於頻繁的檢測
    except KeyboardInterrupt:
        print("shutdown")
    finally:
        GPIO.cleanup()

