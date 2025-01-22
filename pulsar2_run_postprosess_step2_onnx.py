import numpy as np
import cv2
from dataclasses import dataclass

# Constants from the original code
CONFIDENCE_THRESHOLD = 0.45
NMS_THRESHOLD = 0.45
INPUT_SIZE = (640, 640)
REG_MAX = 16

# COCOクラス名
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
    'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]



@dataclass
class Object:
    bbox: list
    label: int
    prob: float

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def decode_distributions(feat, reg_max=16):
    prob = softmax(feat, axis=-1)
    dis = np.sum(prob * np.arange(reg_max), axis=-1)
    return dis

# Copy the complete postprocess function from the original code
# Note: This is the same function from your provided code
def postprocess(outputs, original_shape, input_size, confidence_threshold, nms_threshold, reg_max=16):
    heads = [
        {'output': outputs[0], 'grid_size': input_size[0] // 8, 'stride': 8},
        {'output': outputs[1], 'grid_size': input_size[0] // 16, 'stride': 16},
        {'output': outputs[2], 'grid_size': input_size[0] // 32, 'stride': 32}
    ]
    
    detections = []
    num_classes = 80
    bbox_channels = 4 * reg_max
    class_channels = num_classes

    for head in heads:
        output = head['output']
        batch_size, channels, height, width = output.shape
        stride = head['stride']
        
        if batch_size != 1:
            raise ValueError("Currently only batch size 1 is supported")
            
     

        output = np.transpose(output, (0, 2, 3, 1))
        grid_h = height
        grid_w = width


        bbox_part = output[:, :, :, :bbox_channels]
        class_part = output[:, :, :, bbox_channels:]
        
        num_bbox_params = 4
        if bbox_channels != num_bbox_params * reg_max:
            raise ValueError(f"bbox_channels ({bbox_channels}) does not match 4*reg_max ({4 * reg_max})")
            
        try:
            bbox_part = bbox_part.reshape(batch_size, grid_h, grid_w, num_bbox_params, reg_max)
        except ValueError as e:
            print(f"Failed to reshape bbox_part: {e}")
            raise
            
        bbox_part = bbox_part.reshape(grid_h * grid_w, num_bbox_params, reg_max)
        class_part = class_part.reshape(batch_size, grid_h * grid_w, class_channels)
        
        for b in range(batch_size):
            for i in range(grid_h * grid_w):
                h = i // grid_w
                w = i % grid_w
                class_scores = class_part[b, i, :]
                class_id = np.argmax(class_scores)
                class_score = class_scores[class_id]
                box_prob = sigmoid(class_score)
                
                if box_prob < confidence_threshold:
                    continue
                    
                bbox = bbox_part[i, :, :]
                
                dis_left = decode_distributions(bbox[0, :], reg_max)
                dis_top = decode_distributions(bbox[1, :], reg_max)
                dis_right = decode_distributions(bbox[2, :], reg_max)
                dis_bottom = decode_distributions(bbox[3, :], reg_max)
                
                pb_cx = (w + 0.5) * stride
                pb_cy = (h + 0.5) * stride
                
                x0 = pb_cx - dis_left * stride
                y0 = pb_cy - dis_top * stride
                x1 = pb_cx + dis_right * stride
                y1 = pb_cy + dis_bottom * stride
                
                scale_x = original_shape[1] / input_size[0]
                scale_y = original_shape[0] / input_size[1]
                x0 = np.clip(x0 * scale_x, 0, original_shape[1] - 1)
                y0 = np.clip(y0 * scale_y, 0, original_shape[0] - 1)
                x1 = np.clip(x1 * scale_x, 0, original_shape[1] - 1)
                y1 = np.clip(y1 * scale_y, 0, original_shape[0] - 1)
                
                width = x1 - x0
                height = y1 - y0
                
                detections.append(Object(
                    bbox=[float(x0), float(y0), float(width), float(height)],
                    label=int(class_id),
                    prob=float(box_prob)
                ))
                
    if len(detections) == 0:
        return []
        
    boxes = np.array([d.bbox for d in detections])
    scores = np.array([d.prob for d in detections])
    class_ids = np.array([d.label for d in detections])
    
    final_detections = []
    
    unique_classes = np.unique(class_ids)
    for cls in unique_classes:
        idxs = np.where(class_ids == cls)[0]
        cls_boxes = boxes[idxs]
        cls_scores = scores[idxs]
        
        x1_cls = cls_boxes[:, 0]
        y1_cls = cls_boxes[:, 1]
        x2_cls = cls_boxes[:, 0] + cls_boxes[:, 2]
        y2_cls = cls_boxes[:, 1] + cls_boxes[:, 3]
        
        areas = (x2_cls - x1_cls) * (y2_cls - y1_cls)
        order = cls_scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
                
            xx1 = np.maximum(x1_cls[i], x1_cls[order[1:]])
            yy1 = np.maximum(y1_cls[i], y1_cls[order[1:]])
            xx2 = np.minimum(x2_cls[i], x2_cls[order[1:]])
            yy2 = np.minimum(y2_cls[i], y2_cls[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            intersection = w * h
            iou = intersection / (areas[i] + areas[order[1:]] - intersection)
            
            inds = np.where(iou <= nms_threshold)[0]
            order = order[inds + 1]
            
        for idx in keep:
            final_detections.append(Object(
                bbox=cls_boxes[idx].tolist(),
                label=int(cls),
                prob=float(cls_scores[idx])
            ))
            
    return final_detections

def main():
    # Load the .npy files
    output_0 = np.load('sim_outputs/0/_model_23_Concat_output_0.npy')
    output_1 = np.load('sim_outputs/0/_model_23_Concat_1_output_0.npy')
    output_2 = np.load('sim_outputs/0/_model_23_Concat_2_output_0.npy')
    
    image = cv2.imread("input.jpg")


    # Combine outputs into a list
    outputs = [output_0, output_1, output_2]
    
    # Set original_shape (you might need to adjust these values based on your input image)
    original_shape = image.shape[:2]  # (高さ, 幅)
    
    # Process the outputs
    detections = postprocess(
        outputs,
        original_shape,
        INPUT_SIZE,
        CONFIDENCE_THRESHOLD,
        NMS_THRESHOLD,
        REG_MAX
    )
    
    # Print detection results
    #for det in detections:
    #    print(f"Detection: bbox={det.bbox}, label={det.label}, probability={det.prob:.3f}")

    print("===========================\n")

    # 検出結果の可視化
    for det in detections:
        bbox = [round(x, 5) for x in det.bbox]
        score = round(det.prob, 5)
        class_id = det.label
        class_name = COCO_CLASSES[det.label] if det.label < len(COCO_CLASSES) else f"Unknown class {det.label}"
        print(f"Detected: {class_name}: {score} at bbox {bbox}")
        x, y, w, h = map(int, bbox)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, class_name, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    print("===========================\n")



    cv2.imwrite("output.jpg", image)

if __name__ == "__main__":
    main()
