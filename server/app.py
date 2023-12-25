from flask import Flask, Response, make_response, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO, emit
import cv2 as cv
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import os
import random
import numpy as np
from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, 
    coco_detection_yolo_format_val
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import torch
import threading
import time

input_dir = 'seojun-deep/src/assets'
output_dir = 'dataset2'
camera_url = 'http://192.168.0.178:3000/camera/stream'
checkpoints_dir = 'checkpoints/yolo_nas_s'
predict_image = np.zeros((640, 640, 3))
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model = models.get(
    model_name='yolo_nas_s',
    checkpoint_path='checkpoints/yolo_nas_s/RUN_20231205_103445_069801/ckpt_best.pth',
    num_classes=1
).to(device)
max_classes_len = 0
prediction_lst = []
EPOCHS = 50
BATCH_SIZE = 2
WORKERS = 1
model_to_train = 'yolo_nas_s'
CHECKPOINT_DIR = 'checkpoints'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class PredictThread(threading.Thread):
    def run(self):
        global predict_image, prediction_lst
        count = 0
        cap = cv.VideoCapture(camera_url)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break

            count = count + 1
            if count == 10:
                output = model.predict(frame, conf=0.7)
                prediction_lst = output._images_prediction_lst
                count = 0

            if prediction_lst:
                for prediction in list(prediction_lst):
                    for bbox in prediction.prediction.bboxes_xyxy:
                        x1, y1, x2, y2 = bbox
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            predict_image = frame
        

if not os.path.exists(f'{output_dir}/samples'):
    os.makedirs(f'{output_dir}/samples')

def make_folder(output_dir, folder_name):
    if not os.path.exists(f'{output_dir}/{folder_name}'):
        os.makedirs(f'{output_dir}/{folder_name}')
        os.makedirs(f'{output_dir}/{folder_name}/images')
        os.makedirs(f'{output_dir}/{folder_name}/labels')
    print(f'{folder_name} 폴더 생성 완료')

def delete_prev_dataset():
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            os.remove(os.path.join(root, file))
    print('이전 데이터셋 삭제 완료')

def map_function(x):
    if x < 0:
        return 0
    elif x > 640:
        return 640
    else:
        return int(x)

def check_function(x):
    x1, y1, x2, y2 = x
    if x1 < 0 and x2 < 0:
        print('실패')
        return
    if x1 > 640 and x2 > 640:
        print('실패')
        return
    if y1 < 0 and y2 < 0:
        print('실패')
        return
    if y1 > 640 and y2 > 640:
        print('실패')
        return
    print('성공')
    return x

make_folder(output_dir, 'train')
make_folder(output_dir, 'valid')
make_folder(output_dir, 'test')


class BbsDatasetAug:
    def __init__(self, fname, img, classIds, bbs):
        self.fname = fname
        self.img = img
        self.bbs = []
        for bs in bbs:
            x1, y1, x2, y2 = bs
            self.bbs.append(
                BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            )
        self.bbs = BoundingBoxesOnImage(self.bbs, shape=img.shape)
        self.output_dir = 'dataset2'
        self.colors = [(0, 255, 0), (255, 140, 0), (255, 0, 0)]
        self.aug_count = 100
        self.classIds = classIds
        self.seq = iaa.Sequential([  
                    iaa.Fliplr(0.5),
                    iaa.Flipud(0.2),     
                    iaa.Affine(
                    scale=(0.8, 1.2),
                    rotate=(-45, 45),
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.2, 0.2)},
                    cval=(0, 255)
                    ),
                iaa.LinearContrast((0.7, 1.2))
                ], random_order=True)
        
    def aug(self):
        for i in range(self.aug_count):
            img_aug, bbs_aug = self.seq(image=self.img, bounding_boxes=self.bbs)
            ranSample = random.randrange(1, 4)
            if ranSample == 1:
                draw_img = bbs_aug.draw_on_image(img_aug, size=2)
                cv.imwrite(f'{self.output_dir}/samples/sample-{self.fname}-{i}.jpg', draw_img)

            folder_name = 'train'
            if i < self.aug_count // 10:
                folder_name = 'test'
            elif i < self.aug_count // 3:
                folder_name = 'valid'

            filter_bbs = list(map(check_function, bbs_aug.to_xyxy_array()))

            cv.imwrite(f'{output_dir}/{folder_name}/images/{self.fname}-{i}.jpg', img_aug)

            with open(f'{self.output_dir}/{folder_name}/labels/{self.fname}-{i}.txt', 'w') as f:
                for bbi, bb in enumerate(filter_bbs):
                    if bb is not None:    
                        bb = list(map(map_function, bb))
                        x1, y1, x2, y2 = bb
                        x_center = (x1 + int((x2 - x1) / 2)) / 640
                        y_center = (y1 + int((y2 - y1) / 2)) / 640
                        w = (x2 - x1) / 640
                        h = (y2 - y1) / 640
                        f.write(f'{self.classIds[bbi]} ')
                        f.write(f'{x_center} ')
                        f.write(f'{y_center} ')
                        f.write(f'{w} ')
                        f.write(f'{h} ')
                        f.write('\n')



app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

def get_predict():
    while True:
        _, frame = cv.imencode('.jpg', predict_image)
        yield(b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')

@app.route('/model_list', methods=['GET'])
def get_model_list():
    all_items = os.listdir(checkpoints_dir)
    folder_names = [item for item in all_items if os.path.isdir(os.path.join(checkpoints_dir, item))]
    return jsonify(folder_names)

@app.route('/predict', methods=['GET'])
def on_predict():
    return Response(get_predict(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/change_model', methods=['post'])
def changeModel(data):
    print(data)

@socketio.on('connect')
def on_connect():
    print('socketio connect!')

def generate_dataset(data):
    global max_classes_len
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

        if (len(classIds) > max_classes_len):
            max_classes_len = len(classIds)

        bbsDatasetAug = BbsDatasetAug(f'data{index}', img, classIds, bbs)
        bbsDatasetAug.aug()
        emit('percent', (100 / len(data)) * (index + 1))

def create_array(n):
   return [str(i) for i in range(n)]

def callback_function(epoch, step, metrics):
  emit('percent', EPOCHS / epoch * 100)
  print(f"Epoch: {epoch}, Step: {step}, Metrics: {metrics}")

class YoloNasTrain(threading.Thread):
    def run(self):
        print('train!')
        # emit('percent', 0)
        ROOT_DIR = output_dir
        train_imgs_dir = 'train/images'
        train_labels_dir = 'train/labels'
        val_imgs_dir = 'valid/images'
        val_labels_dir = 'valid/labels'
        test_imgs_dir = 'test/images'
        test_labels_dir = 'test/labels'
        classes = create_array(max_classes_len)
        self.dataset_params = {
            'data_dir': ROOT_DIR,
            'train_imgs_dir': train_imgs_dir,
            'train_labels_dir': train_labels_dir,
            'val_imgs_dir': val_imgs_dir,
            'val_labels_dir': val_labels_dir,
            'test_imgs_dir': test_imgs_dir,
            'test_labels_dir': test_labels_dir,
            'classes': classes
        }
        self.train_data = coco_detection_yolo_format_train(
            dataset_params={
                'data_dir': self.dataset_params['data_dir'],
                'images_dir': self.dataset_params['train_imgs_dir'],
                'labels_dir': self.dataset_params['train_labels_dir'],
                'classes': self.dataset_params['classes']
            },
            dataloader_params={
                'batch_size': BATCH_SIZE,
                'num_workers': WORKERS
            }
        )

        self.val_data = coco_detection_yolo_format_val(
            dataset_params={
                'data_dir': self.dataset_params['data_dir'],
                'images_dir': self.dataset_params['val_imgs_dir'],
                'labels_dir': self.dataset_params['val_labels_dir'],
                'classes': self.dataset_params['classes']
            },
            dataloader_params={
                'batch_size': BATCH_SIZE,
                'num_workers': WORKERS
            }
        )

        self.train_params = {
            'silent_mode': False,
            "average_best_models":True,
            "warmup_mode": "linear_epoch_step",
            "warmup_initial_lr": 1e-6,
            "lr_warmup_epochs": 3,
            "initial_lr": 5e-4,
            "lr_mode": "cosine",
            "callback_function": callback_function,
            "cosine_final_lr_ratio": 0.1,
            "optimizer": "Adam",
            "optimizer_params": {"weight_decay": 0.0001},
            "zero_weight_decay_on_bias_and_bn": True,
            "ema": True,
            "ema_params": {"decay": 0.9, "decay_type": "threshold"},
            "max_epochs": EPOCHS,
            "mixed_precision": True,
            "loss": PPYoloELoss(
                use_static_assigner=False,
                num_classes=len(self.dataset_params['classes']),
                reg_max=16
            ),
            "valid_metrics_list": [
                DetectionMetrics_050(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=len(self.dataset_params['classes']),
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.01,
                        nms_top_k=1000,
                        max_predictions=300,
                        nms_threshold=0.7
                    )
                ),
                DetectionMetrics_050_095(
                    score_thres=0.1,
                    top_k_predictions=300,
                    num_cls=len(self.dataset_params['classes']),
                    normalize_targets=True,
                    post_prediction_callback=PPYoloEPostPredictionCallback(
                        score_threshold=0.01,
                        nms_top_k=1000,
                        max_predictions=300,
                        nms_threshold=0.7
                    )
                )
            ],
            "metric_to_watch": 'mAP@0.50:0.95'
        }

        trainer = Trainer(
            experiment_name=model_to_train, 
            ckpt_root_dir=CHECKPOINT_DIR
        )
        model = models.get(
            model_to_train, 
            num_classes=len(self.dataset_params['classes']), 
            pretrained_weights="coco"
        ).to(DEVICE)

        trainer.train(
            model=model, 
            training_params=self.train_params, 
            train_loader=self.train_data, 
            valid_loader=self.val_data
        )

def train_dataset():
    yolo_nas_train = YoloNasTrain()
    yolo_nas_train.start()

@socketio.on('train')
def on_train(json_str):
    emit('step', 0)
    delete_prev_dataset()
    emit('step', 1)
    generate_dataset(json_str['data'])
    emit('step', 2)
    train_dataset()
    time.sleep(1)
    emit('step', 3)

predict_thread = PredictThread()
# predict_thread.start()

socketio.run(app, port=5000, host='0.0.0.0')




