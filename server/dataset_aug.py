import random
import cv2 as cv
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def check_function(x):
    x1, y1, x2, y2 = x
    if x1 < 0 and x2 < 0:
        return
    if x1 > 640 and x2 > 640:
        return
    if y1 < 0 and y2 < 0:
        return
    if y1 > 640 and y2 > 640:
        return
    return x

def map_function(x):
    if x < 0:
        return 0
    elif x > 640:
        return 640
    else:
        return int(x)

class DatasetAug:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir
        self.colors = [(0, 255, 0), (255, 140, 0), (255, 0, 0)]
        self.aug_count = 100
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
        
    def aug(self, fname, image, classIds, bbs):
        boxes = []
        for bs in bbs:
            x1, y1, x2, y2 = bs
            boxes.append(
                BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
            )
        bounding_boxes = BoundingBoxesOnImage(boxes, shape=image.shape)
        for i in range(self.aug_count):
            ima_aug, bbs_aug = self.seq(image=image, bounding_boxes=bounding_boxes)
            ranSample = random.randrange(1, 4)
            if ranSample == 1:
                draw_ima = bbs_aug.draw_on_image(ima_aug, size=2)
                cv.imwrite(f'{self.dataset_dir}/samples/sample-{fname}-{i}.jpg', draw_ima)

            folder_name = 'train'
            if i < self.aug_count // 10:
                folder_name = 'test'
            elif i < self.aug_count // 3:
                folder_name = 'valid'

            filter_bbs = list(map(check_function, bbs_aug.to_xyxy_array()))

            cv.imwrite(f'{self.dataset_dir}/{folder_name}/images/{fname}-{i}.jpg', ima_aug)

            with open(f'{self.dataset_dir}/{folder_name}/labels/{fname}-{i}.txt', 'w') as f:
                for bbi, bb in enumerate(filter_bbs):
                    if bb is not None:    
                        bb = list(map(map_function, bb))
                        x1, y1, x2, y2 = bb
                        x_center = (x1 + int((x2 - x1) / 2)) / 640
                        y_center = (y1 + int((y2 - y1) / 2)) / 640
                        w = (x2 - x1) / 640
                        h = (y2 - y1) / 640
                        f.write(f'{classIds[bbi]} ')
                        f.write(f'{x_center} ')
                        f.write(f'{y_center} ')
                        f.write(f'{w} ')
                        f.write(f'{h} ')
                        f.write('\n')

