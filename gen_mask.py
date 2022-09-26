from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import argparse
import os
import torch
from loguru import logger
import cv2
from pathlib import Path 
import numpy as np

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Convert VIA dataset into Customdataset for mmsegmentation")
    parser.add_argument("config", 
                        help="config file.")
    parser.add_argument("weight", 
                        help="pretrained weight file in pth form."),
    parser.add_argument("input",
                        help="Specify the input image location."),
    parser.add_argument("--output", default='outputs/yolo',
                        help="Specify the folder location to save segmentation.")
    parser.add_argument("--save",
                        action='store_true',
                        default=False,
                        help='whether to save output binary image files')

    args = parser.parse_args()
    return args

def segment(model, img_file):
    # build the model from a config file and a checkpoint file
    img = mmcv.imread(img_file) 
    mask = inference_segmentor(model, img)
    palette = model.show_result(img, mask, opacity=0.5)
    return mask[0], palette

def group(mask, sensitivity, intact=True):
    '''
    group independent objects.

    Input
    mask: location of segmentation mask (after segmentation process)
    sensitivity : this ratio is the parameter to decide, based on the object size (recommend: 0.0001 ~ 0.01)
    intact: keep it True, if you want to ignore object at the border

    Output
    cropped_masks: a list of cropped object masks
    roi: region of interest for each object mask
    '''
    
    img = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel=(3,3))

    num_groups, _, bboxes, centers  = cv2.connectedComponentsWithStats(img.astype(np.uint8), connectivity=8)
    logger.info(f'The number of {num_groups-1} objects are detected.')
    # bboxes: left most x, top most y, horizontal size, vertical size, total area 
    
    MIN_AREA = img.shape[0] * img.shape[1] * sensitivity 

    cropped_masks = []
    rois = []

    for i, (bbox, center) in enumerate(zip(bboxes, centers)):
        tx, ty, hori, verti, area = bbox
        if area < MIN_AREA or i == 0:
            # skip too small or the whole image
            continue
        roi = ty, ty+verti, tx, tx+hori

        cropped = img[roi[0]:roi[1], roi[2]:roi[3]]

        if cv2.connectedComponents(cropped)[0] != 2: # if there is more than one object in the cropped mask,
            continue

        if intact and any(x in roi for x in [0, img.shape[0], img.shape[1]]): # if a cropped image is located on the image border,
            continue

        cropped_masks.append(cropped)
        rois.append(roi)

    logger.info(f'The number of {len(cropped_masks)} objects are saved.')

    return cropped_masks, rois

def roi2coco(roi, size):
    '''
    convert roi to COCO format.

    Input
    roi: region of interest for each object mask; y left top, y right bottom, x left top, x right bottom.
    size: shape of image; y size, x size

    Output
    coco: normalized xywh format (from 0 to 1)
    '''
    center_x = round((roi[2]+roi[3])/2/size[1], 4)
    center_y = round((roi[0]+roi[1])/2/size[0], 4)
    w_ratio = round((-roi[2]+roi[3])/size[1], 4)
    h_ratio = round((-roi[0]+roi[1])/size[0], 4)
    return center_x, center_y, w_ratio, h_ratio

def generate(type, center_x, center_y, w_ratio, h_ratio, txt_path, CLASS):

    with open(txt_path + '.txt', 'a') as f:
        f.write(('%g ' * 5 + '\n') % (CLASS.index(type), center_x, center_y, w_ratio, h_ratio)) 


def toYolo(mask, txt_name):
    _, rois = group(mask, sensitivity=0.0002)
        
    for roi in rois:

        center_x, center_y, w_ratio, h_ratio = roi2coco(roi, mask.shape)
        generate("plant", center_x, center_y, w_ratio, h_ratio, txt_name, CLASS=["plant"])

if __name__ == '__main__':
    logger.info("Start Hexafarms' Segmentation mask generation.")

    args = parse_args()
    config_file = args.config
    checkpoint_file = args.weight
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_segmentor(config_file, checkpoint_file, device=device)

    if os.path.isdir(args.input):
        logger.info("Input is directory. All images in the directory will be processed.")
        img_location = list(Path(args.input).glob("*.png" or "*.jpg" or "*.jpeg"))

    elif os.path.isfile(args.input):
        logger.info("Input is file. This file will be processed.")
        img_location = [args.input]

    else:
        logger.warn("This is the wrong input.")
    
    i = 1
    j = 1
    cv2.namedWindow("output", cv2.WINDOW_NORMAL)

    while True:
        img_file = img_location[i]

        if i==j:
            logger.info(f"{img_file} is processed.")
            mask, palette = segment(model, img_file)
            j += 1
        
        cv2.imshow("output",palette)
        key = cv2.waitKey(1) & 0xFF
        if key == 32: # Save on Spacebar key
            if args.save:
                logger.info(f"{img_file} is saved at {args.output}")
                mmcv.imwrite(mask, os.path.join(args.output,os.path.basename(img_file)))
            i += 1
            toYolo(mask, os.path.join(args.output,img_file.stem ))

            cv2.destroyAllWindows()
            pass
        elif key == 27: # Close on ESC key
            logger.info(f"This image is not saved.")
            i += 1

            cv2.destroyAllWindows()
            pass



    

