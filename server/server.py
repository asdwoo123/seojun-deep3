from flask import Flask, Response, make_response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import numpy as np
import cv2 as cv
from utilities import delete_prev_dataset, generate_dataset_folder, create_array, searchFolderNames, getModelClasses
import os
# import torch
import sys
# from super_gradients.training import models
# from dataset_aug import DatasetAug
# from yolo_nas_train import YoloNasTrain
# from predict import PredictThread

image_path = '../src/assets/sample.jpg'
image = cv.imread(image_path)
image = cv.resize(image, (640, 640))
predict_image = np.zeros((640, 640, 3))
dataset_dir = 'dataset'
checkpoints_dir = 'checkpoints'
stream_url = 'http://192.168.0.178:3000/camera/stream'
model_name = 'yolo_nas_s'
checkpoint_list = searchFolderNames(f'{checkpoints_dir}/{model_name}')
default_checkpoint = checkpoint_list[0] if len(checkpoint_list) > 0 else None
getModelClasses(checkpoints_dir, model_name, default_checkpoint)

sys.exit()
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
predict_num_classes = 1
predict_model = models.get(
    model_name=model_name,
    checkpoint_path=f'{checkpoints_dir}/{model_name}/{checkpoint_list[0]}/ckpt_best.pth',
    num_classes=predict_num_classes
).to(device) if len(checkpoint_list) > 0 else None

EPOCHS = 100
BATCH_SIZE = 2
WORKERS = 1
train_classes = []

datasetAug = DatasetAug(dataset_dir)
yoloNasTrain = YoloNasTrain(dataset_dir, checkpoints_dir, device)

generate_dataset_folder(dataset_dir)
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# def generate_frame():
#     while True:
#         _, frame = cv.imencode('.jpg', image)
#         yield(b'--frame\r\n'
#                 b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

# @app.route('/stream', methods=['GET'])
# def on_stream():
#     return Response(generate_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/capture', methods=['GET'])
# def on_capture():
#     _, buffer = cv.imencode('.jpg', image)
#     img_str = buffer.tobytes()
#     return Response(img_str, mimetype='image/jpeg')

@app.route('/model_list', methods=['GET'])
def get_model_list():
    folder_names = searchFolderNames(f'{checkpoints_dir}/{model_name}')
    return jsonify(folder_names)

@app.route('/change_model', methods=['post'])
def change_model(name):
    global predict_model
    predict_model = models.get(
        model_name=model_name,
        checkpoint_path=f'checkpoints/{model_name}/{name}/ckpt_best.pth',
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

def generate_dataset(data):
    global train_classes
    for index, label_data in enumerate(data):
        image_buffer, labels = label_data
        img = np.frombuffer(image_buffer, np.uint8)
        img = cv.imdecode(img, cv.IMREAD_COLOR)
        
        bbs = []
        classIds = []
        for label in labels:
            classId, x1, y1, x2, y2 = label
            classIds.append(classId)
            bbs.append(
                [x1, y1, x2, y2]
            )

        if (len(classIds) > len(train_classes)):
            train_classes = classIds

        datasetAug.aug(f'data{index}', img, classIds, bbs)
        emit('percent', (100 / len(data)) * (index + 1))


@socketio.on('train')
def on_train(json):
    emit('step', 0)
    delete_prev_dataset(dataset_dir)
    emit('step', 1)
    generate_dataset(json['data'])
    emit('step', 2)
    yoloNasTrain.train(EPOCHS, BATCH_SIZE, WORKERS, train_classes)
    emit('step', 3)

predict_thread = PredictThread(predict_model, stream_url, predict_image)
predict_thread.daemon = True
predict_thread.start()


socketio.run(app, port=5000, host='0.0.0.0')
