import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as F
from anything_in_anyscene.hdr_sky.sun_models.sunpose_net import *
from anything_in_anyscene.hdr_sky.models.generator import generatorModel
from anything_in_anyscene.hdr_sky.models.discriminator import discriminatorModel
from anything_in_anyscene.hdr_sky.models.vgg16 import NewModel#VGG16
from anything_in_anyscene.hdr_sky.dataset import *
from pathlib import Path
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import pdb
import anything_in_anyscene.hdr_sky.utils as utils
import glob
from json import JSONEncoder
import argparse
import os 

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
torch.manual_seed(0)
np.random.seed(0)

torch.autograd.set_detect_anomaly(True)

# Hyper parameters
# global config
IMSHAPE = (32,128,3)
THRESHOLD = 0.12
HDR_EXTENSION = "hdr" # Available ext.: exr, hdr

FISHEYE_FOV_WIDTH = 120.0 # 120 degree
FISHEYE_FOV_HEIGHT = 90.0 # 90 degree
PANORAMA_FOV_WIDTH = 360.0
PANORAMA_FOV_HEIGHT = 180.0
PANORAMA_IMSHAPE = (64,128,3)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
def dump_peak_vec_to_json(save_path, peak_dir_vec):
    # Serialization
    numpy_data = {"peak_dir": peak_dir_vec}
    print("serialize NumPy array into JSON and write into a file")
    with open(save_path, "w") as write_file:
        json.dump(numpy_data, write_file, cls=NumpyArrayEncoder)
    print("Done writing serialized NumPy array into file")

def read_json_to_peak_vec(load_path):
    # Deserialization
    print("Started Reading JSON file")
    with open(load_path, "r") as read_file:
        print("Converting JSON encoded data into Numpy array")
        decoded_array = json.load(read_file)

        peak_dir_vec = np.asarray(decoded_array["peak_dir"])
        print("NumPy Array peak_dir")
        print(peak_dir_vec)

def calc_peak_dir_vec(peak_position, img_height, img_width, fov_width=360):
    Hs, Ws = img_height, img_width
    fov_rad = fov_width * np.pi / 180.0
    # ys, xs = np.indices((Hs, Ws), np.float32)
    ys, xs = peak_position[0], peak_position[1]

    y_proj = Hs / 2.0 - ys
    x_proj = xs - Ws / 2.0
    theta_alt = x_proj * fov_rad / Ws
    phi_alt = y_proj * np.pi / Hs
    # x = np.sin(theta_alt) * np.cos(phi_alt)
    # y = np.sin(phi_alt)
    # z = np.cos(theta_alt) * np.cos(phi_alt)
    
    x = np.sin(theta_alt) * np.cos(phi_alt)
    y = -np.sin(phi_alt)
    z = np.cos(theta_alt) * np.cos(phi_alt)
    
    # pdb.set_trace()
    return np.array([x, y, z])

def calc_sun_pos_vec(limit_fov_hdr, save_json_path, limit_fov_width=PANORAMA_FOV_WIDTH, limit_fov_height=FISHEYE_FOV_HEIGHT, \
                     panorama_width=128, panorama_height=64):
    ## this function assumes the input ldr has been recovered by network, which has PANORAMA_FOV_WIDTH
    # pdb.set_trace()
    hdr_height, hdr_width, _ = limit_fov_hdr.shape
    pano_img = np.zeros((PANORAMA_IMSHAPE[0],PANORAMA_IMSHAPE[1], 3),dtype=np.float32)
    pano_img[:hdr_height, :,:] = limit_fov_hdr
    # pdb.set_trace()
    ## find peak intensity location (sun location)
    # np_r_img = np.array(pano_img)[:,:,2]
    np_g_img = np.array(pano_img)[:,:,1]
    # np_b_img = np.array(pano_img)[:,:,0]
    peak_pos = unravel_index(np_g_img.argmax(), np_g_img.shape) # (row, col)
    print("pano image peak pos (" + str(peak_pos[0]) + ", " + str(peak_pos[1]) + ")")

    peak_pos_vec = calc_peak_dir_vec(peak_pos, PANORAMA_IMSHAPE[0], PANORAMA_IMSHAPE[1], fov_width=360)
    print("pano image peak vector x:" + str(peak_pos_vec[0]) + ", y:" + str(peak_pos_vec[1]) \
          + ", z:" + str(peak_pos_vec[2]))
    dump_peak_vec_to_json(save_json_path, peak_pos_vec)
    # read_json_to_peak_vec(save_json_path)

def inference(ldr, _sun, _gen, args):
    # pdb.set_trace()
    batch_size, _, h, w,  = ldr.shape
    IMSHAPE = (h,w,_) 

    _sun.eval()
    _gen.eval()
    
    torch.cuda.empty_cache()
    ldr.to(device = args.device)
    """Steps of function generator_in_step"""
    sm, Aks  = _sun(ldr)
    sunpose_pred = torch.reshape(sm, (-1, 1, IMSHAPE[0], IMSHAPE[1]))
    sunlayer1, sunlayer2, sunlayer3 = Aks # [b, h, w, [64, 32, 16]]
    # max_arg = torch.argmax(sun_poses, dim = 1) # max_arg shape (32,), which is the batch size
    # max_arg = max_arg.unsqueeze(dim=-1) ## expand dim to (32, 1)
    # y_c = torch.gather(sm, 1, max_arg)
    y_c,_ = torch.max(sm,1)
    y_c = torch.unsqueeze(y_c,0)
        
    sun_cam1 = utils.gradCamLayer(y_c, sunlayer1)
    sun_cam2 = utils.gradCamLayer(y_c, sunlayer2)
    sun_cam3 = utils.gradCamLayer(y_c, sunlayer3)

    gen_pred = _gen(ldr, sunpose_pred, sun_cam1, sun_cam2, sun_cam3)
    y_final_lin, y_final_gamma = gen_pred[0], gen_pred[1]
    sun_pred_lin = gen_pred[3]
    return y_final_lin, sun_pred_lin

def inference_sky_fov(args, sunpose_m,generator_model, fov=120.0, inpaint=True):
    ldr_imgs = glob.glob(os.path.join(args.indir, '*g'))
    ldr_imgs = sorted(ldr_imgs)
    outdir = args.outdir
    if not os.path.isdir(outdir):
        os.mkdir(outdir)  
        
    img_height, img_width = IMSHAPE[0], IMSHAPE[1]
    img_fov_ratio = fov / 360.0
    img_target_height = img_height
    img_target_width = 128 - 42 * 2 #int(np.floor(img_width * img_fov_ratio))
    img_row_offset = 0
    img_col_offset = 42#(img_width - img_target_width) // 2
    for ldr_img_path in ldr_imgs:
        if not inpaint:
            ldr_img = cv2.imread(ldr_img_path)
            h, w, _ = ldr_img.shape
            dim = (128, 64)
            
            ldr_img = ldr_img[:(ldr_img.shape[0]//2), :]
            
            # insert img to paranoma fov 360
            ldr_img = cv2.resize(ldr_img, (img_target_width, img_target_height))
            l_img = np.zeros((img_height,img_width, 3),dtype=np.float32)
            l_img[img_row_offset:img_row_offset+ldr_img.shape[0], img_col_offset:img_col_offset+ldr_img.shape[1],:] = ldr_img

        else:
            l_img = cv2.imread(ldr_img_path)
            l_img = cv2.resize(l_img, (128, 32))
        
        ldr_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        ldr_val = l_img / 255.0
        ldr_val = torch.unsqueeze(ldr_transform(ldr_val),0).float()
        
        pred_hdr, sun_pred_lin = inference(ldr_val,sunpose_m,generator_model,args)
        # pred_hdr_np = pred_hdr[0].numpy()
        # pdb.set_trace()
        pred_hdr_np = pred_hdr[0].cpu().detach().numpy()
        pred_hdr_np = np.transpose(pred_hdr_np,(1,2,0))
        
        # save_json_path = os.path.join(outdir, file_name_split[0] + '.json')
        save_json_path = os.path.join(outdir, 'sun_peak_dir.json')
        calc_sun_pos_vec(pred_hdr_np, save_json_path, limit_fov_width=PANORAMA_FOV_WIDTH, limit_fov_height=FISHEYE_FOV_HEIGHT, \
                     panorama_width=128, panorama_height=64)
       
        sun_pred_lin_np = sun_pred_lin[0].cpu().detach().numpy()
        sun_pred_lin_np = np.transpose(sun_pred_lin_np,(1,2,0))
        
        cv2.imwrite(os.path.join(outdir, 'sky.hdr'), pred_hdr_np)
        cv2.imwrite(os.path.join(outdir, 'sun_pred' + '.hdr'), sun_pred_lin_np)
        break


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="inference a model")
    parser.add_argument('--indir', type=str, default="/media/sandisk4T_2/lavel_sky_dataset/laval_sky_cut_test_small_entire_net_pytorch/sun_high_intensity_ldr_masked/")
    parser.add_argument('--outdir', type=str, default="/media/sandisk4T_2/lavel_sky_dataset/laval_sky_cut_test_small_entire_net_pytorch/sun_high_intensity_ldr_masked_output/")
    parser.add_argument('--inpaint', default=False, action='store_true', help='whether inpaint')
    
    args = parser.parse_args()

    ## Load all models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    sunpose_m = sunpose_model(im_height=32, im_width= 128)
    
    # ## normal model
    sunpose_checkpoint = torch.load("/anything_in_anyscene/models/hdr_sky/laval_sky_sunpose_epoch1000.pth")
    gen_checkpoint = torch.load("/anything_in_anyscene/models/hdr_sky/laval_sky_generator_epoch1000.pth")

    generator_model = generatorModel(device_list=device)
    discriminator_model = discriminatorModel()
    
    sunpose_m= nn.DataParallel(sunpose_m)
    generator_model= nn.DataParallel(generator_model)
    discriminator_model= nn.DataParallel(discriminator_model)
    
    sunpose_m.to(device)
    generator_model.to(device)
    discriminator_model.to(device)
    
    sunpose_m.load_state_dict(sunpose_checkpoint)
    generator_model.load_state_dict(gen_checkpoint)

    ldr_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # inference our own image
    inference_sky_fov(args, sunpose_m, generator_model, inpaint=args.inpaint)
    print("sky generation finished.")