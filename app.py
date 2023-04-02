import torch
from models.common import DetectMultiBackend
from utils.general import (check_img_size, cv2,
                            non_max_suppression, scale_boxes)
from utils.plots import Annotator, colors
import numpy as np
import gradio as gr
import time
data = 'data/coco128.yaml' 


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']




def detect(im,model,device,iou_threshold=0.45,confidence_threshold=0.25):
    im = np.array(im)
    imgsz=(640, 640)  # inference size (pixels)
    data = 'data/coco128.yaml'  # data.yaml path
    # Load model
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Run inference
    # model.warmup(imgsz=(1))  # warmup
    
    imgs = im.copy()  # for NMS

    image, ratio, dwdh = letterbox(im, auto=False)
    print(image.shape)
    image = image.transpose((2, 0, 1))
    img = torch.from_numpy(image).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

# Inference
    start = time.time()
    pred = model(img, augment=False)
    fps_inference = 1/(time.time()-start)
# NMS
    pred = non_max_suppression(pred, confidence_threshold, iou_threshold, None, False, max_det=10)


    for i, det in enumerate(pred):  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], imgs.shape).round()

            annotator = Annotator(imgs, line_width=3, example=str(names))
            hide_labels = False
            hide_conf = False
            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                print(xyxy,label)
                annotator.box_label(xyxy, label, color=colors(c, True))

    return imgs,fps_inference


def inference(img,model_link,iou_threshold,confidence_threshold):
    print(model_link)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend('weights/'+str(model_link)+'.pt', device=device, dnn=False, data=data, fp16=False)  
    return detect(img,model,device,iou_threshold,confidence_threshold)


def inference2(video,model_link,iou_threshold,confidence_threshold):
    print(model_link)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Load model
    model = DetectMultiBackend('weights/'+str(model_link)+'.pt', device=device, dnn=False, data=data, fp16=False)  
    frames = cv2.VideoCapture(video)
    fps = frames.get(cv2.CAP_PROP_FPS)
    image_size = (int(frames.get(cv2.CAP_PROP_FRAME_WIDTH)),int(frames.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finalVideo = cv2.VideoWriter('output.mp4',cv2.VideoWriter_fourcc(*'VP90'), fps, image_size)
    fps_video = []
    while frames.isOpened():
        ret,frame = frames.read()
        if not ret:
            break
        frame,fps = detect(frame,model,device,iou_threshold,confidence_threshold)
        fps_video.append(fps)
        finalVideo.write(frame)
    frames.release()
    finalVideo.release()
    return 'output.mp4',np.mean(fps_video)



examples_images = ['data/images/bus.jpg',
                    'data/images/zidane.jpg',]
examples_videos = ['data/video/input_0.mp4',
                    'data/video/input_1.mp4'] 

models = ['yolov5s','yolov5n','yolov5m','yolov5l','yolov5x']

with gr.Blocks() as demo:
    gr.Markdown("## YOLOv5 Inference")
    with gr.Tab("Image"):
        gr.Markdown("## YOLOv5 Inference on Image")
        with gr.Row():
            image_input = gr.Image(type='pil', label="Input Image", source="upload")
            image_output = gr.Image(type='pil', label="Output Image", source="upload")
        fps_image = gr.Number(value=0,label='FPS')
        image_drop = gr.Dropdown(choices=models,value=models[0])
        image_iou_threshold = gr.Slider(label="IOU Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.45)
        image_conf_threshold = gr.Slider(label="Confidence Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.25)
        gr.Examples(examples=examples_images,inputs=image_input,outputs=image_output)
        text_button = gr.Button("Detect")
    with gr.Tab("Video"):
        gr.Markdown("## YOLOv5 Inference on Video")
        with gr.Row():
            video_input = gr.Video(type='pil', label="Input Image", source="upload")
            video_output = gr.Video(type="pil", label="Output Image",format="mp4")
        fps_video = gr.Number(value=0,label='FPS')
        video_drop = gr.Dropdown(choices=models,value=models[0])
        video_iou_threshold = gr.Slider(label="IOU Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.45)
        video_conf_threshold = gr.Slider(label="Confidence Threshold",interactive=True, minimum=0.0, maximum=1.0, value=0.25)
        gr.Examples(examples=examples_videos,inputs=video_input,outputs=video_output)
        video_button = gr.Button("Detect")
    
    with gr.Tab("Webcam Video"):
        gr.Markdown("## YOLOv5 Inference on Webcam Video")
        gr.Markdown("Coming Soon")

    text_button.click(inference, inputs=[image_input,image_drop,
                                         image_iou_threshold,image_conf_threshold],
                                        outputs=[image_output,fps_image])
    video_button.click(inference2, inputs=[video_input,video_drop,
                                           video_iou_threshold,video_conf_threshold],            
                                        outputs=[video_output,fps_video])

demo.launch()