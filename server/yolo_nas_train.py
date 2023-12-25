from super_gradients.training import Trainer
from super_gradients.training import dataloaders
from super_gradients.training.dataloaders.dataloaders import (
    coco_detection_yolo_format_train, coco_detection_yolo_format_val 
)
from super_gradients.training import models
from super_gradients.training.losses import PPYoloELoss
from super_gradients.training.metrics import (
    DetectionMetrics_050,
    DetectionMetrics_050_095
)
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
import torch
import utilities

def create_array(n):
   return [str(i) for i in range(n)]

def callback_function(epoch, step, metrics, em, EPOCHS):
  em.emit('percent', EPOCHS / epoch * 100)
  print(f"Epoch: {epoch}, Step: {step}, Metrics: {metrics}")

class YoloNasTrain():
    def __init__(self, dataset_dir, checkpoints_dir, EPOCHS, BATCH_SIZE, WORKERS, max_classes_len):
        self.em = utilities.get_event_bus()
        self.em.emit('percent', 0)
        self.model_to_train = 'yolo_nas_s'
        self.CHECKPOINT_DIR = checkpoints_dir
        ROOT_DIR = dataset_dir
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def train(self):
        trainer = Trainer(
            experiment_name=self.model_to_train,
            ckpt_root_dir=self.CHECKPOINT_DIR
        )

        model = models.get(
            self.model_to_train, 
            num_classes=len(self.dataset_params['classes']), 
            pretrained_weights="coco"
        ).to(self.DEVICE)
        
        trainer.train(
            model=model, 
            training_params=self.train_params, 
            train_loader=self.train_data, 
            valid_loader=self.val_data
        )



