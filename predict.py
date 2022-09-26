from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import argparse
import os
import torch
from pathlib import Path



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
    parser.add_argument("--output", default='outputs',
                        help="Specify the folder location to save segmentation.")
    parser.add_argument("--save",
                        action='store_true',
                        help='whether to save output result files')
    parser.add_argument("--binary",
                        action='store_true',
                        help='whether to save output in binary form')


    args = parser.parse_args()
    return args

def segment(config_file, checkpoint_file, args, device='cuda:0' ):
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device=device)
    
    if args.save:
        if not os.path.exists(args.output):
            os.mkdir(args.output)

        if os.path.isdir(args.input):
            inputs = Path(args.input).glob("*.png")
            for input in inputs:

                img = mmcv.imread(input) 
                result = inference_segmentor(model, img)

                if args.binary:
                    mmcv.imwrite(result[0], os.path.join(args.output, os.path.basename(input)))
                    print('Saved at ', os.path.join(args.output, os.path.basename(input)))
                    # return result
                else:
                    model.show_result(img, result, out_file=os.path.join(args.output, os.path.basename(input)), opacity=0.5)
                    print('Saved at ', os.path.join(args.output, os.path.basename(input)))
                    # return result

        else: #only one image file.
            img = mmcv.imread(args.input) 
            result = inference_segmentor(model, img)
            if args.binary:
                mmcv.imwrite(result[0], os.path.join(args.output, os.path.basename(args.input)))
                print('Saved at ', os.path.join(args.output, os.path.basename(args.input)))
                # return result
            else:
                model.show_result(img, result, out_file=os.path.join(args.output, os.path.basename(args.input)), opacity=0.5)
                print('Saved at ', os.path.join(args.output, os.path.basename(args.input)))
                # return result

def segment_api(config_file, checkpoint_file, input_dir, device='cuda:0' ):
    # build the model from a config file and a checkpoint file
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = init_segmentor(config_file, checkpoint_file, device=device)
    img = mmcv.imread(input_dir)
    mask = inference_segmentor(model, img)
    palette = model.show_result(img, mask, out_file=None, opacity=0.5)
    return mask, palette


'''
Current docker is made only for MMsegmentaiton.
But it should be able to cover fastapi, too.
'''

if __name__ == '__main__':

    args = parse_args()
    config_file = args.config
    checkpoint_file = args.weight
    segment(config_file, checkpoint_file, args)

    

