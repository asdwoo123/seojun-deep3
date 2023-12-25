from flask import Flask, Response, make_response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import cv2 as cv
import threading
import time
import utilities
import os
import torch
from super_gradients.training import models
from dataset_aug import DatasetAug
from yolo_nas_train import YoloNasTrain


image_path = '../src/assets/sample.jpg'
image = cv.imread(image_path)
image = cv.resize(image, (640, 640))
predict_image = np.zeros((640, 640, 3))
dataset_dir = 'dataset'
checkpoints_dir = 'checkpoints'
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = models.get(
    model_name='yolo_nas_s',
    checkpoint_path='checkpoints/yolo_nas_s/RUN_20231205_103445_069801/ckpt_best.pth',
    num_classes=1
).to(device)
EPOCHS = 100
BATCH_SIZE = 2
train_classes = []
predict_classes = []

datasetAug = DatasetAug(dataset_dir)
yoloNasTrain = YoloNasTrain(dataset_dir, checkpoints_dir)

utilities.generate_dataset_folder(dataset_dir)
em = utilities.get_event_bus()
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

def generate_frame():
    while True:
        _, frame = cv.imencode('.jpg', image)
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

@app.route('/stream', methods=['GET'])
def on_stream():
    return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['GET'])
def on_capture():
    _, buffer = cv.imencode('.jpg', image)
    img_str = buffer.tobytes()
    return Response(img_str, mimetype='image/jpeg')

@app.route('/model_list', methods=['GET'])
def get_model_list():
    all_items = os.listdir(checkpoints_dir)
    folder_names = [item for item in all_items if os.path.isdir(os.path.join(checkpoints_dir, item))]
    return jsonify(folder_names)

@app.route('/change_model', methods=['post'])
def change_model(model_name):
    model = models.get(
        model_name='yolo_nas_s',
        checkpoint_path=f'checkpoints/yolo_nas_s/${model_name}/ckpt_best.pth',
        num_classes=1
    ).to(device)

def get_predict():
    while True:
        _, frame = cv.imencode('.jpg', predict_image)
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

@app.route('/predict')
def on_predict():
    return Response(get_predict(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def socket_connect():
    print('socketio connect!')

@socketio.on('train')
def on_train(json):
    em('step', 0)
    utilities.delete_prev_dataset()
    em('step', 1)

    

socketio.run(app, port=5000, host='0.0.0.0')
