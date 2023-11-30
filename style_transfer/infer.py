import os
import random
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import Image #, ImageFilter
from anything_in_anyscene.style_transfer.model.networks import Generator
from anything_in_anyscene.style_transfer.utils.tools import get_config, get_model_list #, random_bbox, mask_image, is_image_file, default_loader, normalize
from anything_in_anyscene.pytorch_render.corner_case.render.util import foreground_and_noise
import skimage

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="training configuration")
parser.add_argument('--seed', type=int, help='manual seed')
parser.add_argument('--input', type=str)
parser.add_argument('--flow', type=str, default='')
parser.add_argument('--iter', type=int, default=0)

def main():
    args = parser.parse_args()
    config = get_config(args.config)
    checkpoint_path = config["infer_checkpoint_path"]
    if not os.path.exists(checkpoint_path):
        raise ValueError("no right checkpoint path specified")
    # CUDA configuration
    cuda = config['cuda']
    device_ids = config['gpu_ids']
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)
        device_ids = list(range(len(device_ids)))
        config['gpu_ids'] = device_ids
        cudnn.benchmark = True
    print(f"Arguments: {args}")

    # Set random seed
    if args.seed is None:
        args.seed = random.randint(1, 10000)
    print(f"Random seed: {args.seed}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed_all(args.seed)

    print(f"Configuration: {config}")
    # Define the trainer
    netG = Generator(config['netG'], cuda, device_ids)
    # Resume weight
    last_model_name = get_model_list(checkpoint_path, "gen", iteration=args.iter)
    netG.load_state_dict(torch.load(last_model_name))
    if cuda:
        netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
    # model_iteration = int(last_model_name[-11:-3])
    print(f"Resume from {last_model_name}")
    try:  # for unexpected error logging
        with torch.no_grad():   # enter no grad context
            # for clip_name in sorted(os.listdir(args.input)):
                # if not os.path.isdir(os.path.join(args.input, clip_name)):
                #     continue
            # clip_path = os.path.join(args.input, clip_name)
            content_path = os.path.join(args.input, 'pre_result')
            mask_path = os.path.join(args.input, 'mask')
            # obj_path = os.path.join(args.input, 'pre_result')
            # raw_path = os.path.join(clip_path, 'raw')

            for img_name in os.listdir(content_path):
                if not img_name.endswith('.png'):
                    continue
                content_img_path = os.path.join(content_path, img_name)
                hardmask_img_path = os.path.join(mask_path, "Hard"+img_name)
                # obj_img_path = os.path.join(obj_path, img_name)
                # raw_img_path = os.path.join(raw_path, img_name)
            
                fullsize_content_ori = Image.open(content_img_path).convert("RGB")
                fullsize_hardmask = Image.open(hardmask_img_path).split()[3] #only need alpha channel
                fullsize_hardmask_arr = np.expand_dims(np.array(fullsize_hardmask), axis=-1)

                fullsize_content_arr = np.array(fullsize_content_ori)
                fullsize_obj_arr = fullsize_content_arr*(fullsize_hardmask_arr/255.0)
                fullsize_obj = Image.fromarray(fullsize_obj_arr.astype('uint8'))

                # fullsize_content_arr[(fullsize_hardmask_arr!=0).any(axis=-1)] = 0
                fullsize_content_arr = fullsize_content_arr*(1.0 - fullsize_hardmask_arr/255.0)
                fullsize_content = Image.fromarray(fullsize_content_arr.astype('uint8'))

                seg_pixels = np.where(np.any(fullsize_hardmask_arr !=0, axis=-1))
                seg_umin = np.min(seg_pixels[1])
                seg_vmin = np.min(seg_pixels[0])
                seg_umax = np.max(seg_pixels[1])
                seg_vmax = np.max(seg_pixels[0])

                crop_umin, crop_umax, crop_vmin, crop_vmax = seg_umin, seg_umax, seg_vmin, seg_vmax
                suaqred_box_size = max([crop_umax - crop_umin + 1, crop_vmax - crop_vmin + 1])*1.1
                u_center = int((crop_umin + crop_umax)/2)
                v_center = int((crop_vmin + crop_vmax)/2)

                crop_umin = max(0, u_center - int(suaqred_box_size/2))
                crop_vmin = max(0, v_center - int(suaqred_box_size/2))
                crop_umax = min(np.shape(fullsize_content)[1] - 1, u_center + int(suaqred_box_size/2)+1)
                crop_vmax = min(np.shape(fullsize_content)[0] - 1, v_center + int(suaqred_box_size/2)+1)
                crop_content = fullsize_content.crop((crop_umin, crop_vmin, crop_umax, crop_vmax))
                crop_mask = fullsize_hardmask.crop((crop_umin, crop_vmin, crop_umax, crop_vmax))
                crop_obj = fullsize_obj.crop((crop_umin, crop_vmin, crop_umax, crop_vmax))
                resize_input_tf = transforms.Resize((960,960))

                rs_crop_content = resize_input_tf(crop_content)
                rs_crop_mask = resize_input_tf(crop_mask)
                rs_crop_obj = resize_input_tf(crop_obj)

                x = transforms.ToTensor()(rs_crop_content)
                mask = transforms.ToTensor()(rs_crop_mask)[0].unsqueeze(dim=0)
                obj = transforms.ToTensor()(rs_crop_obj)
                # x = normalize(x)
                # x = x * (1. - mask)
                x = x.unsqueeze(dim=0)
                mask = mask.unsqueeze(dim=0)
                # obj = normalize(obj)
                obj = obj.unsqueeze(dim=0)

                if cuda:
                    # netG = nn.parallel.DataParallel(netG, device_ids=device_ids)
                    x = x.cuda()
                    mask = mask.cuda()
                    obj = obj.cuda()

                # Inference
                x1, x2, offset_flow = netG(x, mask, obj)
                # inpainted_result = x2 * mask + x * (1. - mask)
                inpainted_result = torch.clamp(x2,0,1)
                resize_tf = transforms.Resize((crop_vmax-crop_vmin, crop_umax-crop_umin))
                resized_trans_patch = resize_tf(inpainted_result.cpu()).squeeze()
                replace_patch = (resized_trans_patch * 255).numpy().astype('uint8').transpose((1, 2, 0)) #convert to (height, width, channel)
                # replace_patch = cv2.medianBlur(replace_patch, 3)
                replace_patch = skimage.util.random_noise(replace_patch/255.0, mode="speckle", var=0.003)*255
                replace_patch = replace_patch.astype('uint8')
                post_result = np.array(fullsize_content)
                post_result[crop_vmin:crop_vmax, crop_umin:crop_umax] = replace_patch
                # post_result[np.all(fullsize_hardmask_arr ==0, axis=-1)] = np.array(fullsize_content)[np.all(fullsize_hardmask_arr ==0, axis=-1)]
                post_result = (1-fullsize_hardmask_arr/255.0)*np.array(fullsize_content_ori) + fullsize_hardmask_arr/255.0*post_result # HxWxC
                post_result = torch.tensor((post_result.transpose((2,0,1)))) #CxHxW
                # output_dir = os.path.join(args.input, os.path.basename(checkpoint_path))
                output_dir = os.path.join(args.input, "post_result")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                output_path = os.path.join(output_dir, img_name)
                vutils.save_image(post_result, output_path, padding=0, normalize=True)
                # crop_content.save(os.path.join(output_dir, img_name.split('.')[0]+"content.png"))
                # crop_mask.save(os.path.join(output_dir, img_name.split('.')[0]+"mask.png"))
                # crop_obj.save(os.path.join(output_dir, img_name.split('.')[0]+"obj.png"))
                print(f"Saved the result to {output_path}")
                if args.flow:
                    vutils.save_image(offset_flow, args.flow, padding=0, normalize=True)
                    print(f"Saved offset flow to {args.flow}")
                                    # exit no grad context
    except Exception as e:  # for unexpected error logging
        print(f"Error: {e}")
        raise e


if __name__ == '__main__':
    main()
