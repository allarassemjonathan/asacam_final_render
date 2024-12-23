#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
from flask import Flask, Response, render_template, jsonify, request, redirect, url_for, send_file
import cv2
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import argparse
import os
import time
from loguru import logger
import cv2
import torch
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

app = Flask(__name__)

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]
global_queue_content = []
global_queue_frames = []

def arti_parser(video_path):
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="video", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default='yolo_l')
    parser.add_argument("-n", "--name", type=str, default='yolox-l', help="model name")

    parser.add_argument(
        "--path", default=f"{video_path}", help="path to video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    parser.add_argument(
        "--app", 
        default="run"
    )
    parser.add_argument(
        "--host=0.0.0.0",
        default=""
    )
    return parser


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "demo", default="image", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    parser.add_argument(
        "--path", default="./assets/dog.jpg", help="path to images or video"
    )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default=None,
        type=str,
        help="please input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="cpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=None, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
    # model_dir = "C:\\Users\\ALLARASSEMJJ20\\productx\\tools\\vit-gpt2-image-captioning"
    # # Directory containing the pytorch_model.bin and config files

    # # Load the model using the directory where the .bin file is stored
    # model = VisionEncoderDecoderModel.from_pretrained(model_dir)

    # # Load the feature extractor and tokenizer
    # feature_extractor = ViTImageProcessor.from_pretrained(model_dir)
    # tokenizer = AutoTokenizer.from_pretrained(model_dir)

        # if trt_file is not None:
        #     from torch2trt import TRTModule

        #     model_trt = TRTModule()
        #     model_trt.load_state_dict(torch.load(trt_file))

        #     x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
        #     self.model(x)
        #     self.model = model_trt

    # def inference(self, img):
    #     img_info = {"id": 0}
    #     if isinstance(img, str):
    #         img_info["file_name"] = os.path.basename(img)
    #         img = cv2.imread(img)
    #     else:
    #         img_info["file_name"] = None

    #     height, width = img.shape[:2]
    #     img_info["height"] = height
    #     img_info["width"] = width
    #     img_info["raw_img"] = img

    #     ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
    #     img_info["ratio"] = ratio

    #     img, _ = self.preproc(img, None, self.test_size)
    #     img = torch.from_numpy(img).unsqueeze(0)
    #     img = img.float()
    #     if self.device == "gpu":
    #         img = img.cuda()
    #         if self.fp16:
    #             img = img.half()  # to FP16

    #     with torch.no_grad():
    #         t0 = time.time()
    #         outputs = self.model(img)
    #         if self.decoder is not None:
    #             outputs = self.decoder(outputs, dtype=outputs.type())
    #         outputs = postprocess(s
    #             outputs, self.num_classes, self.confthre,
    #             self.nmsthre, class_agnostic=True
    #         )
    #         #logger.info("Infer time: {:.4f}s".format(time.time() - t0))
    #     return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


import json
import requests
import base64
import logging
from fpdf import FPDF
from datetime import datetime

# Configure logger
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%H:%M:%S')

# OpenAI API key = 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
api_key ="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"
# Function to encode the image
def encode_image(frame):
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return None, {"error": "Failed to encode image"}
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8'), None

# function to send the frame to VIT-GPT2
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

# import os
# # Assuming `vit-gpt2-image-captioning` is located in the current working directory.
# relative_model_path = r"vit-gpt2-image-captionings"
# # Get the absolute path
# model_dir_gpt2_vit = os.path.abspath(relative_model_path)
# print("Absolute Path: ", absolute_model_path)

# Directory containing the pytorch_model.bin and config files
#model_dir_gpt2_vit1 ="C:\\Users\\ALLARASSEMJJ20\\productx\\tools\\vit-gpt2-image-captioning"

# model_dir_gpt2_vit = model_dir_gpt2_vit.replace('\\', '\\\\')

# # Load the model using the directory where the .bin file is stored
# gpt2_vit = VisionEncoderDecoderModel.from_pretrained(model_dir_gpt2_vit, use_auth_token=False)

# # Load the feature extractor for vit gpt2
# feature_extractor_gtp2_vit = ViTImageProcessor.from_pretrained(model_dir_gpt2_vit, use_auth_token=False)

# # Load the tokenizer for vit gpt2
# tokenizer_gtp2_vit = AutoTokenizer.from_pretrained(model_dir_gpt2_vit, use_auth_token=False)

# def interpret_frame_with_vit(frame)->str:
#     # Load an example image
#     # Replace with your image path
#     #image = Image.open(frame).convert("RGB")# Ensure the image is in RGB format
#     image = Image.fromarray(frame).convert("RGB")

#     # Preprocess the image
#     inputs = feature_extractor_gtp2_vit(images=image, return_tensors="pt")

#     # Generate caption
#     output_ids = gpt2_vit.generate(inputs["pixel_values"], max_length=16, num_beams=4)

#     # Decode the generated IDs to text
#     caption = tokenizer_gtp2_vit.decode(output_ids[0], skip_special_tokens=True)

#     return caption

def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if args.save_result:
        save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time))
        os.makedirs(save_folder, exist_ok=True)
        
        save_path = os.path.join(save_folder, os.path.basename(args.path) if args.demo == "video" else "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    
    frame_count = 0
    INTERPRET_FRAME_INTERVAL = 2  # interpret every 5th frame
    collected_logs = []
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            
            if args.save_result:
                vid_writer.write(result_frame)
           # else:
                #cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                #cv2.imshow("yolox", result_frame)
            
            frame_count += 1
            #collected_content = []
            if frame_count % INTERPRET_FRAME_INTERVAL == 0:
                content = interpret_frame_with_vit(result_frame)
                logger.info(f"{content}")
                current_time = datetime.now().strftime('%H:%M:%S')
                #log_message = f"{current_time}: {content}"
                log_message = (current_time, content)
                collected_logs.append(log_message)
                #collected_content.append(content)
                #logger.info(f"Frame {frame_count} content: {content}")
                
                # Split text into multiple lines based on width
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_color = (255, 255, 255)  # White color text
                thickness = 1
                line_type = cv2.LINE_AA

                window_width = frame.shape[1]
                max_line_width = window_width - 20  # 10 pixels padding on each side
                words = content.split(' ')
                lines = []
                current_line = words[0]
                '''
                for word in words[1:]:
                    # Check the width of the current line + the next word
                    text_size, _ = cv2.getTextSize(current_line + ' ' + word, font, font_scale, thickness)
                    if text_size[0] < max_line_width:
                        current_line += ' ' + word
                    else:
                        lines.append(current_line)
                        current_line = word
                lines.append(current_line)  # Add the last line

                # Calculate the size of the entire text box
                line_height = cv2.getTextSize("A", font, font_scale, thickness)[0][1]
                box_height = len(lines) * (line_height + 5) + 10  # 5 pixels padding between lines
                box_width = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in lines]) + 10  # 10 pixels padding for text width

                # Background rectangle coordinates
                x, y = window_width - box_width - 10, 20  # 10 pixels padding from right and top
                background_top_left_corner = (x, y)
                background_bottom_right_corner = (x + box_width, y + box_height)

                # Drawing the background rectangle
                cv2.rectangle(result_frame, background_top_left_corner, background_bottom_right_corner, (0, 0, 0), -1)

                # Putting text on the rectangle
                for i, line in enumerate(lines):
                    text_y = y + (i + 1) * (line_height + 5)
                    cv2.putText(result_frame, line, (x + 5, text_y), font, font_scale, font_color, thickness, line_type)
                
                window_name = "Special Frame Display"
                cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow(window_name, frame)
                cv2.waitKey(10000)  # Wait for 20 seconds
                cv2.destroyWindow(window_name)
                
            # Display the results with overlay
            if args.save_result:
                vid_writer.write(result_frame)
            else:            
                cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                cv2.imshow("yolox", result_frame)
            '''
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break
    pdf = PDF()
    pdf.add_page()
    pdf.add_content(collected_logs)
    pdf_output_path = 'interpreted_frames_report.pdf'
    pdf.output(pdf_output_path)
    print(f"PDF report saved as {pdf_output_path}")


def generate_frames(video_source):
    global camera
    camera = cv2.VideoCapture(video_source)  # Open video stream

    while True:
        success, frame = camera.read()  # Read frame from the video
        if not success:
            break
        else:
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            # Yield the frame in a format suitable for streaming


            # detection


            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# Configure logger
logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s: %(message)s',
                    datefmt='%H:%M:%S')

# OpenAI API key = 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

# Function to encode the image
def encode_image(frame):
    success, encoded_image = cv2.imencode('.jpg', frame)
    if not success:
        return None, {"error": "Failed to encode image"}
    return base64.b64encode(encoded_image.tobytes()).decode('utf-8'), None

prompt = ""
# Function to send the frame to OpenAI API
def interpret_frame_with_openai(frame):
    global prompt
    base64_image, error = encode_image(frame)
    if error:
        return error
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}. Be concise." 
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()
    else:
        logger.error(f"Failed to interpret image: {response.text}")
        return {"error": response.text, "status_code": response.status_code}


def image_flow_demo_textgen_UI_integrated(predictor, vis_folder, current_time, args):
    global global_queue_content
    print('path', args.path)
    print('demo',  args.demo)
    cap = cv2.VideoCapture(args.path if args.demo == "run" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    INTERPRET_FRAME_INTERVAL = 5  # interpret every 5th frame
    collected_logs = []
    while True:
        print('here1')
        ret_val, frame = cap.read()
        if not ret_val:
            cap = cv2.VideoCapture(args.path if args.demo == "run" else args.camid)
            cap.set()
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = 0
            INTERPRET_FRAME_INTERVAL = 10  # interpret every 5th frame
            collected_logs = []
        if ret_val:
            print('here1*')
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            
           # else:
                #cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                #cv2.imshow("yolox", result_frame)
            
            frame_count += 1
            #collected_content = []
            if frame_count % INTERPRET_FRAME_INTERVAL == 0:
                print('here2')
                current_time = datetime.now().strftime('%H:%M:%S')
                content = interpret_frame_with_vit(result_frame)
                global_queue_content.append(content)
                logger.info(f"{content}")
                #log_message = f"{current_time}: {content}"
                log_message = (current_time, content)
                collected_logs.append(log_message)
                #collected_content.append(content)
                #logger.info(f"Frame {frame_count} content: {content}")
                
                # Split text into multiple lines based on width
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_color = (255, 255, 255)  # White color text
                thickness = 1
                line_type = cv2.LINE_AA

                window_width = frame.shape[1]
                max_line_width = window_width - 20  # 10 pixels padding on each side
                words = content.split(' ')
                lines = []
                current_line = words[0]

                yield json.dumps({"text": f"{current_time} :: Camera {lst_sources[args.path]} \n {content}"})

    # pdf = PDF()
    # pdf.add_page()
    # pdf.add_content(collected_logs)
    # pdf_output_path = 'interpreted_frames_report.pdf'
    # pdf.output(pdf_output_path)
    # print(f"PDF report saved as {pdf_output_path}")

contacted = False
def image_flow_demo_openai_UI_integrated(predictor, vis_folder, current_time, args):
    global contacted
    global global_queue_content
    global ind
    print('path', args.path)
    print('demo',  args.demo)
    # if the path has changed aka args.path is not video_source
    path = args.path
    cap = cv2.VideoCapture(path if args.demo == "run" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    frame_count = 0
    INTERPRET_FRAME_INTERVAL = 10 # interpret every 5th frame
    collected_logs = []
    while True:
        print('here1')
        ret_val, frame = cap.read()
        if not ret_val:
            print('restarting LLM')
            cap = cv2.VideoCapture(path if args.demo == "run" else args.camid)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)

        if ret_val:
            print('here1*')
            # outputs, img_info = predictor.inference(frame)
            # result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            
           # else:
                #cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                #cv2.imshow("yolox", result_frame)
            
            frame_count += 1
            #collected_content = []
            if frame_count % INTERPRET_FRAME_INTERVAL == 0:
                print('here2')
                content = interpret_frame_with_openai(frame)['choices'][0]['message']['content']
                if 'ALERT' in content:
                    # email
                    print('about to send')
                    email_send_dest(your_email, recipient_email, content)
                    contacted = True

                current_time = datetime.now().strftime('%H:%M:%S')
                global_queue_content.append(f"{current_time} :: Camera {lst_sources[args.path]} ::\n {content}")
                logger.info(f"{content}")
                #log_message = f"{current_time}: {content}"
                log_message = (current_time, content)
                collected_logs.append(log_message)
                #collected_content.append(content)
                #logger.info(f"Frame {frame_count} content: {content}")

                print('values', args.path, video_source)
                if args.path != video_source:
                    break
                print('sources', lst_sources, args.path, video_source)
                yield json.dumps({"text": f"{current_time} :: Camera {lst_sources[video_source]} ::\n {content}"})


def imageflow_demo_yolo_UI_integrated(predictor, vis_folder, current_time, args):

    print('path', args.path, 'demo', args.demo)
    cap = cv2.VideoCapture(args.path if args.demo == "run" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print('step1 ok')
    # if args.save_result:
    #     print('saving the frames')
    #     save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M", current_time))
    #     os.makedirs(save_folder, exist_ok=True)
        
    #     save_path = os.path.join(save_folder, os.path.basename(args.path) if args.demo == "video" else "camera.mp4")
    #     logger.info(f"video save_path is {save_path}")
        
    #     vid_writer = cv2.VideoWriter(
    #         save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    #     )
    
    if not os.path.exists(args.path):
        print('path does not exist')
    else:
        print('victory')

    if not cap.isOpened():
        print('path exist but cannot be opened')
    else:
        print('victoryy')

    print('about to start')
    while True:
        print('I am reading')
        ret_val, frame = cap.read()

        if not ret_val:
            print('restarting CV')
            print('path', args.path, 'demo', args.demo)
            cap = cv2.VideoCapture(args.path if args.demo == "run" else args.camid)
            
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
            fps = cap.get(cv2.CAP_PROP_FPS)
            # if args.save_result:
            #     print('saving the frames')
            #     save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M", current_time))
            #     os.makedirs(save_folder, exist_ok=True)
                
            #     save_path = os.path.join(save_folder, os.path.basename(args.path) if args.demo == "video" else "camera.mp4")
            #     logger.info(f"video save_path is {save_path}")
                
            #     vid_writer = cv2.VideoWriter(
            #         save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
            #     )
            
        if ret_val:
            # outputs, img_info = predictor.inference(frame)
            # result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            # print('i am reading inside')
            # if args.save_result:
            #     vid_writer.write(result_frame)

            # ret, buffer = cv2.imencode('.jpg', result_frame)

            # frame = buffer.tobytes()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
    
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        

LOGO_PATH = 'logo_white.jpg'
LOGO_WIDTH = 10  # Adjust as needed
class PDF(FPDF):
   def header(self):
       self.set_font('Courier', 'B', 12)
       self.cell(0, 10, 'productx Mission Report', 0, 1, 'C')
       # Add logos in the corners of the header
       self.image(LOGO_PATH, 10, 8, LOGO_WIDTH)  # Top-left corner
       self.image(LOGO_PATH, 190, 8, LOGO_WIDTH)  # Top-right corner

   def footer(self):
       self.set_y(-15)
       self.set_font('Courier', 'I', 8)
       self.image(LOGO_PATH, 10, self.get_y(), LOGO_WIDTH)  # Bottom-left corner
       self.image(LOGO_PATH, 190, self.get_y(), LOGO_WIDTH)  # Bottom-right corner
       self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

   def add_content(self, content_list):
       self.set_font('Courier', '', 12)
       for timestamp, content in content_list:
            self.set_font('Courier', 'B', 12)
            self.cell(30, 10, timestamp, ln=0)  # Keeping the timestamp width fixed
            self.set_font('Courier', '', 12)
            self.multi_cell(0, 10, f" {content}")
'''
   def add_content(self, content_list):
       self.set_font('Courier', '', 12)
       for timestamp, content in content_list:
            self.set_font('Courier', 'B', 12)
            self.cell(30, 10, timestamp, ln=0)  # Keeping the timestamp width fixed
            self.set_font('Courier', '', 12)
            self.multi_cell(0, 10, f" {content}")
'''
# if __name__ == "__main__":
#    args = make_parser().parse_args()
#     print(args)
#     exp = get_exp(args.exp_file, args.name)
#     print(exp)
#     main(exp, args)


# Initialize the camera
camera = None

# Function to generate random text
def generate_random_text():
    texts = ["Jonathan is moving", "Jonathan touched his hair", "Jonathan is staring hard at the screen"]
    return texts[0]

# @app.route('/mission')
# def index():
#     return render_template('index.html')

# @app.route('/')
# def home():
#     return render_template('home.html')

def LLM_text_pipeline(predictor, vis_folder, current_time, args)->str:
    # run the text model
    return Response(image_flow_demo_textgen_UI_integrated(predictor, vis_folder, current_time, args),
                    content_type='text/plain')

video_source = None
lst_sources = {}
ind=0
current_time = None
@app.route('/video_feed')
def video_feed():
    global video_source
    global current_time
    global ind

    video_source = request.args.get('url')  # Get the URL from the query parameters
    predictor = None
    # if not video_source:
    #     video_source = "tools\\static\\videoplayback.mp4"
    # read the path from here and pass it into the args
    args = arti_parser(video_source).parse_args()
    exp = get_exp(args.exp_file, args.name)
    print('test', args.path, exp.output_dir, args.experiment_name)
    
    # keep a list of all the new cameras added to the system
    if args.path not in lst_sources:
        lst_sources[args.path]=ind
        ind+=1

    # process the rtsp link and create a directory for it
    # the goal of the directory is to make sure we can filter out
    # the null path which does not seem to get filter out when checking
    # if args.path == None (bizzare)

    # clean_path = args.path
    # if 'rtsp' in clean_path:

    #     # if the path is "good" we add make a dir so
    #     # that we access the CV if it is triggered.

    #     clean_path = args.path.replace('/', '_')
    #     clean_path = clean_path.replace(':', '_')
    #     clean_path = clean_path.replace('%', '_')
    #     os.makedirs(clean_path, exist_ok=True)

    # while not os.path.exists(clean_path):
    #     redirect(url_for('video_feed'))

    print("passing with", args.path)
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    print('saving or not', args.save_result)
    if args.save_result:
        print('saving...')
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()
    
    # if not args.trt:
    #     if args.ckpt is None:
    #         ckpt_file = os.path.join(file_name, "best_ckpt.pth")
    #     else:
    #         ckpt_file = args.ckpt
    #     logger.info("loading checkpoint")
    #     ckpt = torch.load(ckpt_file, map_location="cpu")
    #     # load the model state dict
    #     model.load_state_dict(ckpt["model"])
    #     logger.info("loaded checkpoint done.")

    # if args.fuse:
    #     logger.info("\tFusing model...")
    #     model = fuse_model(model)

    # if args.trt:
    #     assert not args.fuse, "TensorRT model is not support model fusing!"
    #     trt_file = os.path.join(file_name, "model_trt.pth")
    #     assert os.path.exists(
    #         trt_file
    #     ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
    #     model.head.decode_in_inference = False
    #     decoder = model.head.decode_outputs
    #     logger.info("Using TensorRT to inference")
    # else:
    #     trt_file = None
    #     decoder = None

    # predictor = Predictor(
    #     model, exp, COCO_CLASSES, trt_file, decoder,
    #     args.device, args.fp16, args.legacy,
    # )
    # current_time = time.localtime()

    print('starting CV ..... ')

    print(args.path)

    if 'rtsp' not in args.path: 
        while not os.path.exists(args.path):
            redirect(url_for('video_feed'))

    # video_pipeline(predictor, vis_folder, current_time, args)
    # # LLM_text_pipeline(predictor, vis_folder, current_time, args)

    # thread1 = threading.Thread(target=video_pipeline, args=(predictor, vis_folder, current_time, args))
    # thread2 = threading.Thread(target=LLM_text_pipeline, args=(copy.deepcopy(predictor), copy.deepcopy(vis_folder), copy.deepcopy(current_time), copy.deepcopy(args)))
    
    # thread1.start()
    # thread2.start()

    # thread1.join()
    # thread2.join()

    return Response(imageflow_demo_yolo_UI_integrated(predictor, vis_folder, current_time, args),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# the solo purpose of this route is to reinitialze
# the global variable first to True

@app.route('/load_stream', methods=['POST'])
def load_stream():
    global first
    global contacted
    # helpful to restart this so 
    # that I can reload the stream
    contacted = True
    first = True
    print('restarting the LLM cycle')
    # You could include additional validation for the URL here if needed
    return jsonify(success=True)

# Set up the SMTP server
smtp_server = "smtp.gmail.com"
smtp_port = 587
your_email = "jonathanjerabe@gmail.com"
your_password = "ajrn mros lkzm urnu"
recipient_email = ""

@app.route('/prompt', methods=['POST'])
def get_prompt():
    global prompt
    global recipient_email
    data = request.get_json()
    recipient_email = data.get('email')
    prompt = data.get('prompt')
    print(f"prompt received {prompt}")
    if len(prompt)>0:
        return jsonify(success=True)
    return jsonify(success=False)

def email_send_dest(sender, dest, content):

    # Compose the email
    subject = "ASACAM REPORT"
    body = content
    print('hereemail')
    # Create the MIME message
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = dest
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        # Establish connection to Gmail's SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection

        # Log in to the server
        server.login(your_email, your_password)

        # Send the email
        text = msg.as_string()
        server.sendmail(your_email, recipient_email, text)

        print("Email sent successfully!")

    except Exception as e:
        print(f"Error sending email: {e}")

    finally:
        # Close the connection to the server
        server.quit()


@app.route('/email', methods=['POST'])
def email_reporter():
    data = request.get_json()

    # Extract the stream URL, content, and email from the received data
    content = data.get('content')
    email = data.get('email')
    reporter = data.get('reporter')
    title = data.get('title')

    # Compose the email
    recipient_email = email
    subject = "ASACAM REPORT"
    body = f'Hey {reporter},\n\n\nAn alarm has been triggered among the cameras that you own for the mission {title}. Below is the current report:'+ content + '\n\n\n' + 'we recommend you check the application as soon as possible.\n\nASACAM AUTOMATED SERVICE TEST\n\n'

    # Create the MIME message
    msg = MIMEMultipart()
    msg['From'] = your_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    # Connect to the SMTP server and send the email
    try:
        # Establish connection to Gmail's SMTP server
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()  # Secure the connection

        # Log in to the server
        server.login(your_email, your_password)

        # Send the email
        text = msg.as_string()
        server.sendmail(your_email, recipient_email, text)

        print("Email sent successfully!")

    except Exception as e:
        print(f"Error sending email: {e}")

    finally:
        # Close the connection to the server
        server.quit()
    # You could include additional validation for the URL here if needed
    return jsonify(success=True)


# Route to get random text dynamically // THIS IS GONNA BE THE LLM TEXT
first = True
predictor_LLM = None
vis_folder_LLM = None
args_LLM = None
old_source = None
@app.route('/random_text')
def random_text():

    # these values are global because i not only need to initialize
    # them outside the loop, i also need them to conserve the state 
    # of the function.
    global old_source
    global video_source
    global first
    global predictor_LLM
    global vis_folder_LLM
    global args_LLM
    global global_queue_content
    print('1xxxx')
    print(video_source)
    if video_source==None:
        redirect(url_for('random_text'))
    print('first', first)
    if first:
        print('ok check?')
        old_source = video_source
        # if not video_source:
        #     video_source = "tools\\static\\videoplayback.mp4"
        # read the path from here and pass it into the args
        print('2xxxx')
        args_LLM = arti_parser(video_source).parse_args()
        exp = get_exp(args_LLM.exp_file, args_LLM.name)
        print('xxxx', args_LLM.path)

        if 'rtsp' not in args_LLM.path:
            while not os.path.exists(args_LLM.path):
                redirect(url_for('video_feed'))

        if not args_LLM.experiment_name:
            args_LLM.experiment_name = exp.exp_name

        file_name = os.path.join(exp.output_dir, args_LLM.experiment_name)
        os.makedirs(file_name, exist_ok=True)

        if args_LLM.save_result:
            vis_folder_LLM = os.path.join(file_name, "vis_res")
            os.makedirs(vis_folder_LLM, exist_ok=True)

        if args_LLM.trt:
            args_LLM.device = "gpu"

        logger.info("Args: {}".format(args_LLM))

        if args_LLM.conf is not None:
            exp.test_conf = args_LLM.conf
        if args_LLM.nms is not None:
            exp.nmsthre = args_LLM.nms
        if args_LLM.tsize is not None:
            exp.test_size = (args_LLM.tsize, args_LLM.tsize)

        model = exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

        if args_LLM.device == "gpu":
            model.cuda()
            if args_LLM.fp16:
                model.half()  # to FP16
        model.eval()
        predictor_LLM = None
        # if not args_LLM.trt:
        #     if args_LLM.ckpt is None:
        #         ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        #     else:
        #         ckpt_file = args_LLM.ckpt
        #     logger.info("loading checkpoint")
        #     ckpt = torch.load(ckpt_file, map_location="cpu")
        #     # load the model state dict
        #     model.load_state_dict(ckpt["model"])
        #     logger.info("loaded checkpoint done.")

        # if args_LLM.fuse:
        #     logger.info("\tFusing model...")
        #     model = fuse_model(model)

        # if args_LLM.trt:
        #     assert not args_LLM.fuse, "TensorRT model is not support model fusing!"
        #     trt_file = os.path.join(file_name, "model_trt.pth")
        #     assert os.path.exists(
        #         trt_file
        #     ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        #     model.head.decode_in_inference = False
        #     decoder = model.head.decode_outputs
        #     logger.info("Using TensorRT to inference")
        # else:
        #     trt_file = None
        #     decoder = None

        # predictor_LLM = Predictor(
        #     model, exp, COCO_CLASSES, trt_file, decoder,
        #     args_LLM.device, args_LLM.fp16, args_LLM.legacy,
        # )
        current_time_LLM = time.localtime()
        old_source = video_source
        first=False
        return Response(image_flow_demo_openai_UI_integrated(predictor_LLM, vis_folder_LLM, current_time_LLM, args_LLM))
    
    # if source has changed and it is not the first time
    # 
    if old_source!=video_source:
        first = True
        print('second chance')
        return random_text()
  
    if global_queue_content:
        return Response(json.dumps({"text":global_queue_content[-1]}))
    return Response(json.dumps({"text":"Loading Image Analysis ... "}))

# Folder to save uploaded PDFs

import glob 

# get the last created directory
def get_last_created_directory(path):
    # Get all subdirectories in the specified path
    subdirs = [d for d in glob.glob(os.path.join(path, '*')) if os.path.isdir(d)]

    # If there are no directories, return None
    if not subdirs:
        return None

    # Sort subdirectories by creation time (newest first)
    latest_subdir = max(subdirs, key=os.path.getctime)
    
    print(latest_subdir)
    return latest_subdir

# Define the path to the 'vis_res' directory
VIS_RES_FOLDER = './YOLOX_outputs\yolo_l\\vis_res\\'

@app.route('/old_missions')
def old_missions():
    # Get all subdirectories in 'vis_res'
    # vis_res_folders = [f for f in os.listdir(VIS_RES_FOLDER) if os.path.isdir(os.path.join(VIS_RES_FOLDER, f))]
    # print('folders', vis_res_folders)
    # # For each folder, find the report and video
    # resources = []
    # for folder in vis_res_folders:
    #     folder_path = os.path.join(VIS_RES_FOLDER, folder)
    #     report = None
    #     video = None
        
    #     # Search for report and video in each folder
    #     for file in os.listdir(folder_path):
    #         if file.endswith('.pdf'):
    #             report = file
    #         elif file.endswith('.mp4'):
    #             video = file
    #         print('file', file)
    #     # Add to resources list if both report and video are found
    #     if report and video:
    #         resources.append({
    #             'folder': folder,
    #             'report': report,
    #             'video': video
    #         })

    print('got here kinda')
    resources = [('tools\YOLOX_outputs\yolo_l\\vis_res\\2024_10_20_03_25\\camera.mp4','tools\YOLOX_outputs\yolo_l\\vis_res\\2024_10_20_03_25\\report.pdf'), '']
    return render_template('old_missions.html', resources=resources)


@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    global args_LLM
    if 'pdf' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['pdf']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # check if the file exist
        # put the right there
        # Save the file
        
        exp = get_exp(args_LLM.exp_file, args_LLM.name)
        if exp==None:
            return jsonify({'message': 'Error', 'file_path':'None'}), 500
        if not args_LLM.experiment_name:
            args_LLM.experiment_name = exp.exp_name
        file_name = os.path.join(exp.output_dir, args_LLM.experiment_name)
        vis_folder = os.path.join(file_name, "vis_res")
        current_time = time.localtime()
        print(vis_folder)
        print('last path guy', get_last_created_directory(vis_folder))
        save_folder = os.path.join(vis_folder, time.strftime("%Y_%m_%d_%H_%M", current_time))
        
        # if the image folder is already created use that instead of creating a new folder
        image_folder = get_last_created_directory(vis_folder)
        if save_folder != image_folder and image_folder!=None:
            save_folder = image_folder
            print('used the older folder')

        if os.path.exists(save_folder):
            print('I think this is it', save_folder)
        else:
            print('Does not exist oops...')
            os.makedirs(save_folder, exist_ok=True)

        save_folder = os.path.join(save_folder, 'report.pdf')
        print('final folder', save_folder)
        file.save(save_folder)
        return jsonify({'message': 'PDF saved successfully!', 'file_path':save_folder}), 200

@app.route('/vis_res/<path:folder>/<path:filename>')
def serve_file(folder, filename):
    print('test', folder, filename)
    path = os.path.join(VIS_RES_FOLDER, folder)
    path = os.path.join(path, filename)
    print(path)
    return send_file(path, as_attachment=False, conditional=False)


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/mission')
def index():
    return render_template('mission.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/activity')
def reports():
    return render_template('activity.html')

@app.route('/alerts')
def alerts():
    return render_template('alerts.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

if __name__ == '__main__':
    app.run(host="0.0.0.0")