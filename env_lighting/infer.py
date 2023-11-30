import torch
import torch.nn as nn
import argparse
import json
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from anything_in_anyscene.env_lighting.core.Data_provider import build_dataset
from anything_in_anyscene.env_lighting.core.utils import avg_psnr, save_fig, AverageMeter
import glob
import os
import os.path
from os import path
from PIL import Image
import imageio
import numpy as np
import cv2

import pdb

parser = argparse.ArgumentParser()


#model
parser.add_argument('--model_name', type=str, default='CEVR_NormNoAffine_Maps')
parser.add_argument('--decode_name', type=str, default='mult_resizeUp_map')
parser.add_argument('--act', type=str, default='leaky_relu')
parser.add_argument('--mlp_num', type=int, default=3)
parser.add_argument('--pretrain', type=str, default='vgg')
parser.add_argument('--cycle', action='store_true', default=False)
parser.add_argument('--sep', type=float, default=0.5)
parser.add_argument('--EV_info',  type=int, default=1, help="1: only cat dif, 2: cat source and dif, 3: Embed DIF to 16 dim vec")
parser.add_argument('--init_weight',  action='store_true', default=False)
parser.add_argument('--norm_type', type=str, default='GroupNorm', help="LayerNorm, GroupNorm, InstanceNorm") 
parser.add_argument('--NormAffine', action='store_true', default=False)

# dataset
parser.add_argument('--data_root', type=str, default='/home/skchen/HDR_research/HDREye/images/LDR/')
parser.add_argument('--Float_Stack1', action='store_true', default=False)
parser.add_argument('--Float_Stack2', action='store_true', default=False)
parser.add_argument('--Float_Stack3', action='store_true', default=False)
parser.add_argument('--photomatix_path', type=str, default="/all/CEVR/PhotomatixCL")

# exp path
#parser.add_argument('--exp_path', type=str, default='./train_strategy/experiment/Standard_noLNAffine_Whole/') # Exp folder
parser.add_argument('--B_model_path', type=str, default='CEVR_NormNoAffine_Maps_GN_Bmodel') # Exp folder
parser.add_argument('--D_model_path', type=str, default='CEVR_NormNoAffine_Maps_GN_Dmodel') # Exp folder
parser.add_argument('--save_path', type=str, default='/all/CEVR/results') # Exp folder

parser.add_argument('--resize', action='store_true', default=False)
parser.add_argument('--epoch', type=str, default='620') # Exp folder

args = parser.parse_args()

save_path = args.save_path
exp_base = "/anything_in_anyscene/models/env_lighting/"
D_path = exp_base + args.D_model_path
B_path = exp_base + args.B_model_path


if args.resize:
	print("!!!!!!!!!!inference on 256*256")
	transform = transforms.Compose([
	    transforms.Resize((256, 256)),
	    transforms.ToTensor()
	])
else:
	print("!!!!!!!!!!inference on original size")
	transform = transforms.Compose([
	    transforms.ToTensor()
	])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Initializing with device:", device)

# Set up dataset info
data_path = args.data_root # '/home/skchen/Research_HDR_hunlin/HDREye/images/LDR/'

scene_fold = []
for folder_name in ["cam3", "cam4", "cam5", "cam6"]:
    scene_name = os.path.join(data_path, folder_name)
    scene_fold.append(scene_name)

print("scene_fold: ", scene_fold)


exp_fold_int = [-3, -2, -1, 1, 2, 3]
if args.Float_Stack1:
	exp_fold_float = [-3, -2.5 ,-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 2.5, 3]
	exp_fold = exp_fold_float
	print("Generating Floating EV stack1")

elif args.Float_Stack2:
	exp_fold_float = [-3, -2, -1.5, -1.25, -1, -0.5, 0.5, 1, 1.25, 1.5, 2, 3]
	exp_fold = exp_fold_float
	print("Generating Floating EV stack2")

elif args.Float_Stack3:
	exp_fold_float = [-3, -2.5, -2, -1.5, -1.25, -1, -0.5, 0.5, 1, 1.25, 1.5, 2, 2.5, 3]
	exp_fold = exp_fold_float
	print("Generating Floating EV stack3")

else:
	exp_fold = exp_fold_int
	print("Generating Integer EV stack")


print("Dataset info preparation!!")

# Build up output image folder
#save_path = args.exp_path + "exp_result_HDREye_" + "epoch" + args.epoch + '/'


if path.exists(save_path) == False:
    print("makedir: ", save_path )
    os.makedirs(save_path)
else:
    print("exp_result folder: ", save_path , " existed!")


# Build up inc/dec model and load weight
if args.cycle:
    from anything_in_anyscene.env_lighting.core.cycle_model import build_network
    print('cycle model')
    model = build_network(args)
else:
    from anything_in_anyscene.env_lighting.core.HDR_model import build_network
    print('normal model')
    model_inc = build_network(args)
    model_dec = build_network(args)

"""
if args.best == False:
	model_inc.load_state_dict(torch.load(args.exp_path + 'inc/final_model.pth'))
	model_inc.to(device)
	model_dec.load_state_dict(torch.load(args.exp_path + 'dec/final_model.pth'))
	model_dec.to(device)
	print("Final Model build up and load weight successfully!!")
else:
	model_inc.load_state_dict(torch.load(args.exp_path + 'inc/model_best.pth'))
	model_inc.to(device)
	model_dec.load_state_dict(torch.load(args.exp_path + 'dec/model_best.pth'))
	model_dec.to(device)
	print("Best Model build up and load weight successfully!!")
"""
# weight_name = 'model_' + args.epoch + '.pth'
weight_name = 'final_model.pth'
model_inc.load_state_dict(torch.load(os.path.join(B_path, 'inc', weight_name)))
model_inc.to(device)
model_dec.load_state_dict(torch.load(os.path.join(D_path, 'dec', weight_name)))
model_dec.to(device)
print("Model build up and load weight successfully!!", " Weight name: ", weight_name)




# inference
with torch.no_grad():
	model_inc.eval()
	model_dec.eval()

	for scene in scene_fold:
		save_list = []

		print("Processing Scene: ", scene)
		# build up scene folder in exp_result
		scene_path = os.path.join(save_path, scene.split('/')[-1])     #'./train_strategy/experiment/milestone2-1/exp_result_HDREye/C35'
		if path.exists(scene_path) == False:
			print("makedir: ", scene_path)
			os.makedirs(scene_path)

		# Get source image
		#EV_zero_img_path = data_path + scene+ "/" + scene+ "_0EV_true.jpg.jpg"
		EV_zero_img_path = sorted(glob.glob(os.path.join(scene,'*g')))[0]

		#pdb.set_trace()


		EV_zero_img = transform(Image.open(EV_zero_img_path).convert('RGB')).unsqueeze(0).to(device)

		for tar_exp in exp_fold:
			#print("tar_exp= ", tar_exp)

			# Get ground truth image
			"""
			if tar_exp in exp_fold_int:			
				gt_path = data_path + scene+ "/" + scene + "_" + str(tar_exp) + "EV_true.jpg.jpg"
				gt = transform(Image.open(gt_path).convert('RGB')).unsqueeze(0).to(device)
			"""

			step = torch.tensor([0 + tar_exp], dtype=torch.float32).unsqueeze(0).to(device)
			ori = torch.tensor([0], dtype=torch.float32).unsqueeze(0).to(device)

			if tar_exp > 0:
				out = model_inc(EV_zero_img, step, ori)
				#print("inc act")
			if tar_exp < 0:
				out = model_dec(EV_zero_img, step, ori)
				#print("dec act")

			"""
			if tar_exp in exp_fold_int:
				psnr = avg_psnr(out, gt)
				print("Scene ", scene, ", EV ", tar_exp, " PSNR:",psnr)
				ev_dict[str(tar_exp)].update(psnr)
			"""

			out = out.squeeze(0).cpu() # From (bs,c,h,w) back to (c,h,w)
			
			if args.resize:
				output_path = scene_path + "/EV" + str(tar_exp) + ".jpg"
			else:
				output_path = scene_path + "/EV" + str(tar_exp) + "_ori.jpg"
			save_img = save_fig(out, output_path)
			save_list.append(output_path)  

		if args.resize:
			out_zero_path =  scene_path + "/EV0.jpg"
		else:
			output_path = scene_path + "/EV" + str(tar_exp) + "_ori.jpg"
		zero_img = EV_zero_img.squeeze(0).cpu()
		save_img = save_fig(zero_img, out_zero_path)
		save_list.append(out_zero_path) 

		output_name = scene.split("/")[-1]
		hdr_path = os.path.join(save_path, output_name + ".hdr")
		ldr_path = os.path.join(save_path, output_name + ".jpg")
		hdr_name = hdr_path.split(".")[0]
		os.system(f"{args.photomatix_path}/PhotomatixCL-static -allowwatermark -32 -o {hdr_name} "
		f"{save_list[0]} {save_list[1]} {save_list[2]} {save_list[3]} {save_list[4]} "
		f"{save_list[5]} {save_list[6]} {save_list[7]} {save_list[8]} "
		f"{save_list[9]} {save_list[10]} {save_list[11]} {save_list[12]}")

		imageio.plugins.freeimage.download()
		hdr_img = imageio.imread(hdr_path, hdr_path.split(".")[-1])
		hdr_img = np.array(hdr_img)
		print(hdr_img.max())
		gamma = 1
		hdr_img = (hdr_img / 1).astype('float') ** (1 / gamma)       
		ldr_img = (hdr_img * 255).clip(0, 255).astype('uint8')
		cv2.imwrite(ldr_path, ldr_img[:,:,2::-1])


