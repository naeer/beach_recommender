import os
import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ExifTags, ImageOps
from django.conf import settings
from utils.general import (
    check_img_size, non_max_suppression, scale_coords
)
from utils.torch_utils import select_device
from utils.plots import Annotator, colors
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from django.conf import settings
from django.contrib.sessions.backends.db import SessionStore
from cam_app.database_operations import *
import datetime



def detector(model, image, img_size=640, conf_thres=0.20, iou_thres=0.40, max_det=1000, line_thickness=3):
    device = model.device
    
    # Prepare the image
    img = letterbox(image, img_size, stride=model.stride, auto=True)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)

    # Convert image to torch tensor
    img_tensor = torch.from_numpy(img).to(device)
    img_tensor = img_tensor.half() if model.fp16 else img_tensor.float()
    img_tensor /= 255.0
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor[None]

    # Inference
    pred = model(img_tensor)

    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes=None, agnostic=False, max_det=max_det)

    # Process predictions
    im0 = image.copy()
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(im0, line_width=line_thickness, example=str(model.names))

    bbox_count = 0  # Variable to store the number of bounding boxes

    for i, det in enumerate(pred):  # per image
        if len(det):
            bbox_count += len(det)  # Increment the bounding box count

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], im0.shape).round()

            # Draw boxes and labels
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{model.names[c]} {conf:.2f}'
                annotator.box_label(xyxy, label, color=colors(c, True))

    output_image = annotator.result()

    return output_image, bbox_count  # Return the output image and the bounding box count


def get_model_and_running_func(weights_path, data_path, img_size=640):
    device = select_device()
    
    # Load the model
    model = DetectMultiBackend(weights_path, device=device, dnn=False, data=data_path, fp16=False)
    
    # Check and set the image size
    img_size = check_img_size(img_size, s=model.stride)
    
    # Warm up the model
    model.warmup(imgsz=(1, 3, img_size, img_size))
    
    return model, detector


class PreLoadedVideo(object):
    def __init__(self, source):
        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        # self.video = cv2.VideoCapture(0)
        # If you decide to use video.mp4, you must have this file in the folder
        # as the main.py.
        if source == "queenscliff":
            self.video = cv2.VideoCapture('queenscliff_tuesday_2pm.mp4')
        elif source == "compilation":
            self.video = cv2.VideoCapture('compilation_seagull.mp4')

    def __del__(self):
        self.video.release()

    def get_frame_with_detection(self, function_run=None, model=None, threshold=0.20, iou_threshold=0.40):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        output_image, bbox_count = function_run(model, image, conf_thres=threshold, iou_thres=iou_threshold)
        ret, outputImagetoReturn = cv2.imencode('.jpg', output_image)

        return outputImagetoReturn.tobytes(), output_image, bbox_count  # Return the output image, encoded frame, and bounding box count
    
    def get_frame_without_detection(self):
        success, image = self.video.read()
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        outputs = image
        outputImage = outputs
        ret, outputImagetoReturn = cv2.imencode('.jpg', outputImage)
        return outputImagetoReturn.tobytes(), outputImage


def generate_frames(camera, AI, source):
    frame_count = 0
    bbox_counts = []
    bbox_average = None
    bbox_dict = {}

    try:
        while True:
            if AI:
                 
                threshold = float(settings.CONF_THRESHOLD)
                print("model video threshold: " + str(threshold))
                model, function = get_model_and_running_func(weights_path=str(os.path.join(settings.WEIGHTS_DIR, 'best.pt')), data_path=str(os.path.join(settings.DATA_DIR, 'new_data_yaml_ger.yaml')))
                frame, img, bbox_count = camera.get_frame_with_detection(function, model, threshold, iou_threshold=float(settings.IOU_THRESHOLD))
                bbox_counts.append(bbox_count)

                # Save the frame count and bounding box count in the dictionary
                bbox_dict[frame_count] = bbox_count

                if len(bbox_counts) == 2:
                    bbox_average = int(sum(bbox_counts) / len(bbox_counts))
                    print("Average Bounding Box Count:", bbox_average)
                    bbox_counts = []
                    if source == "compilation":
                        table_name = "detection_level_compilation"
                    elif source == "queenscliff":
                        table_name = "detection_level_queenscliff"
                    create_table('data.db', table_name, 'people_count TEXT, crowdedness TEXT, date_time DATETIME')
                    current_datetime = datetime.datetime.now()
                    format_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

                    if bbox_average is not None:
                        if bbox_average > 20:
                            predicted_crowdedness = 'High'
                            print(f"Level of crowdness: {predicted_crowdedness}")
                            insert_data_to_table('data.db', table_name, f'{str(bbox_average)},{predicted_crowdedness},{str(format_datetime)}')
                        else:
                            predicted_crowdedness = 'Low'
                            print(f"Level of crowdness: {predicted_crowdedness}")
                            insert_data_to_table('data.db', table_name, f'{str(bbox_average)},{predicted_crowdedness},{str(format_datetime)}')
            if not AI:
                frame, img = camera.get_frame_without_detection()

            frame_count += 1

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            
    except Exception as e:
        print(e)
    finally:
        print("Reached finally, detection stopped")
        cv2.destroyAllWindows()

