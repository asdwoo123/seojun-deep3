import threading
import cv2 as cv

class PredictThread(threading.Thread):
    def __init__(self, model, stream_url, predict_image):
        self.model = model
        self.predict_image = predict_image
        self.prediction_lst = []
        self.cap = cv.VideoCapture(stream_url)        

    def run(self):
        count = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to read frame")
                break

            count += 1
            if count == 10:
                output = self.model.predict(frame, conf=0.7)
                self.prediction_lst = output._images_prediction_lst
                count = 0

            if self.prediction_lst:
                for prediction in list(self.prediction_lst):
                    for bbox in prediction.prediction.bboxes_xyxy:
                        x1, y1, x2, y2 = bbox
                        x1 = int(x1)
                        y1 = int(y1)
                        x2 = int(x2)
                        y2 = int(y2)
                        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.predict_image = frame


            