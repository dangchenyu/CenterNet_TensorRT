import os
import cv2
import time
import math
import numpy as np
import tensorrt as trt
from xavier_demo_det_utils import *

# ------------------- parameters -------------------
# global parameters
flag_time_static = True
flag_show_result = True
flag_cameral = False
flag_video = False

#
ind_capture_device = 1
size_cap = [1280, 720]  # width height
# cap = cv2.VideoCapture(0)  #rec_20191119_165440.avi
cap = cv2.VideoCapture('M19040313550500131.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, size_cap[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size_cap[1])
time_program_total = 0
time_total_detection = 0
time_total_action = 0
time_total_pose = 0
ind_frame = 0
start_ind_frame_want_to_test = 0

# detection parameters
# OUTPUT_NAME = ["conv_45", "conv_53", "conv_61", "conv_69"]
model_detection = detection_block(
    deploy_file='/home/ubilab/Source/Taxi-demand-prediction/caffe_models/DLASeg.prototxt',
    model_file='/home/ubilab/Source/Taxi-demand-prediction/caffe_models/DLASeg.caffemodel',
    engine_file='/home/ubilab/Source/Taxi-demand-prediction/caffe_models/DLASeg.engine',
    input_shape=(3, 512, 512),
    output_name=['conv_blob53', 'conv_blob55', 'conv_blob57'],
    data_type=trt.float32,
    flag_fp16=True,
    max_workspace_size=1,
    max_batch_size=1,
    num_class=1,
    max_per_image=20,
    vis_thresh=0.5)
process_caffemodel = process_caffemodel(model_detection)
context_detection, h_input_detection, d_input_detection, h_output_detection, d_output_detection, stream_detection = process_caffemodel.get_outputs()
while True:
    print(ind_frame)
    ind_frame += 1
    ret, frame = cap.read()
    if not ret:
        break
    frame = frame[80:, :640]
    start_time = time.time()
    model_detection.process_det_frame(frame=frame, pagelocked_buffer=h_input_detection)

    model_detection.do_inference(context_detection, h_input_detection, d_input_detection, h_output_detection, d_output_detection,
                 stream_detection)
    output_box_detection = model_detection.posprocess_detection(h_output_detection, frame)
    end_time = time.time()
    print('total', end_time - start_time)