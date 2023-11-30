import os
import torch
torch.manual_seed(0)
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import json
import cv2
import anything_in_anyscene.hdr_sky.utils as utils
import pdb
import torchvision.transforms as transforms
from numpy import unravel_index
os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4, 5"

PI = np.math.pi
HDR_EXTENSION = "hdr" # Available ext.: exr, hdr
IMSHAPE = (32,128,3)
AZIMUTH_gt = IMSHAPE[1]*0.5-1

def sunpose_init(i, h, w):
    # xy coord to degree
    # gap value + init (half of the gap value)

    x = ((i+1.) - np.floor(i/w) * w - 1.) * (360.0/w) + (360.0/(w*2.)) 
    y = (np.floor(i/w)) * (90./h) + (90./(2.*h))

    # deg2rad
    phi = (y) * (PI / 180.)
    theta = (x - 180.0) * (PI / 180.)

    # rad2xyz
    x_u = np.cos(phi) * np.cos(theta)
    y_u = np.sin(phi)
    z_u = np.cos(phi) * np.sin(theta)
    p_u = [x_u, y_u, z_u]
    
    # return np.array(p_u)
    return p_u

SUNPOSE_BIN = [sunpose_init(i,IMSHAPE[0],IMSHAPE[1]) for i in range(IMSHAPE[0]*IMSHAPE[1])]

def mean_std_for_loader1(loader: DataLoader):
  mean = torch.zeros(3)
  std = torch.zeros(3)
  mean_peak_intensity = torch.zeros(3)
  std_peak_intensity = torch.zeros(3)
  mean_peak_dir = torch.zeros(3)
  std_peak_dir = torch.zeros(3)
  # pdb.set_trace()
  for (ldr_imgs, peak_intensities, peak_dirs, peak_dir_map, peak_intensity_map, hdr_imgs) in loader:
    # print('peak int')
    # print(peak_intensities.shape)
    # print('peak peak_dirs')
    # print(peak_dirs.shape)
    for d in range(3):
      mean[d] += ldr_imgs[:, d, :, :].mean()
      std[d] += ldr_imgs[:, d, :, :].std()
      mean_peak_intensity[d] += peak_intensities[:, d].mean()
      std_peak_intensity[d] += peak_intensities[:, d].std()
      mean_peak_dir[d] += peak_dirs[:, d, :, :].mean()
      std_peak_dir[d] += peak_dirs[:, d, :, :].std()
  mean.div_(len(loader))
  std.div_(len(loader))
  mean_peak_intensity.div_(len(loader))
  std_peak_intensity.div_(len(loader))
  mean_peak_dir.div_(len(loader))
  std_peak_dir.div_(len(loader))
  # return list(mean.numpy()), list(std.numpy())
  return list(mean.numpy()), list(std.numpy()), \
         list(mean_peak_intensity.numpy()), list(std_peak_intensity.numpy()), \
         list(mean_peak_dir.numpy()), list(std_peak_dir.numpy())

def CalcNormParamters(csv_file, root_dir, num_train_sample, num_test_sample):
  ldr_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.ToTensor(),
  ])
  hdr_transform = transforms.Compose([
      transforms.ToTensor()
  ])
  sample_dataset = HDRDataset(csv_file, root_dir, ldr_transform=ldr_transform,\
                           hdr_transform=hdr_transform, peak_intensity_transform=None, peak_dir_transform=None)
  train_set, test_set = torch.utils.data.random_split(sample_dataset, [num_train_sample,num_test_sample]) # split data number [train, test]
  # pdb.set_trace()
  train_loader = DataLoader(dataset = train_set, batch_size = num_train_sample, shuffle = True) # batch size = entire dataset
  # test_loader = DataLoader(dataset = test_set, batch_size = 1, shuffle = True)

  ## compute mean and std for data normalization
  means, stds, mean_peak_intensity, std_peak_intensity, mean_peak_dir, std_peak_dir = mean_std_for_loader1(train_loader)
  print(means)
  print(stds)
  print(mean_peak_intensity)
  print(std_peak_intensity)
  print(mean_peak_dir)
  print(std_peak_dir)

class HDRDataset(Dataset):
  def __init__(self, csv_file, root_dir, ldr_transform=None, hdr_transform=None,\
               peak_intensity_transform=None, peak_dir_transform=None):
    self.annotations = pd.read_csv(csv_file)
    print(len(self.annotations))
    self.root_dir = root_dir
    self.ldr_transform = ldr_transform
    self.hdr_transform = hdr_transform
    self.peak_intensity_transform = peak_intensity_transform
    self.peak_dir_transform = peak_dir_transform
    self.hdr_clip_threshold = 1048576.0 # 2^20
    self.fov = 360.0
    self.img_height = 512
    self.img_width = 2048
    self.pos_encoding_map = self.calc_pos_encoding(self.img_height, self.img_width)

  def vMF(self, x, y, h, w, kappa=80.0):
    # discrete the sun into (h*w) bins and model the sun probability distirbution. (von Mises-Fisher)
    # pdb.set_trace()
    sp_vec = utils.sphere2world((x, y), h, w, skydome=True)
    sp_vec = np.expand_dims(sp_vec, axis=0)
    sp_vec = np.tile(sp_vec, (h*w, 1))
    
    batch_dot = np.einsum("bc, bc-> b", SUNPOSE_BIN, sp_vec)
    batch_dot = batch_dot.astype(np.float32) # convert float64 to float32
    batch_dot = kappa * batch_dot
    pdf = np.exp(batch_dot)
    
    pdf = pdf / np.sum(pdf)
    return pdf

  def calc_dir_vec(self, position, img_height, img_width):
    Hs, Ws = img_height, img_width
    fov_rad = self.fov * np.pi / 180.0
    # ys, xs = np.indices((Hs, Ws), np.float32)
    ys, xs = position[0], position[1]
    # ys, xs = 512, 1024
    # ys, xs = 0, 1100
    y_proj = Hs / 2.0 - ys
    x_proj = xs - Ws / 2.0
    theta_alt = x_proj * fov_rad / Ws
    phi_alt = y_proj * np.pi / Hs
    x = np.sin(theta_alt) * np.cos(phi_alt)
    y = np.sin(phi_alt)
    z = np.cos(theta_alt) * np.cos(phi_alt)
    # pdb.set_trace()
    return np.array([x, y, z])

  def calc_pos_encoding(self, img_height, img_width):
    pos_encoding_map = np.zeros((img_height, img_width, 3), dtype=np.float32)
    for y in range(img_height):
      for x in range(img_width):
        # pdb.set_trace()
        u_vec = self.calc_dir_vec([y, x], img_height, img_width)
        pos_encoding_map[y,x,0] = u_vec[0]
        pos_encoding_map[y,x,1] = u_vec[1]
        pos_encoding_map[y,x,2] = u_vec[2]

    return pos_encoding_map

  def calc_peak_dir_encoding(self, peak_dir_vec, img_height, img_width):
    peak_dir_encoding_map = np.zeros((img_height, img_width, 1), dtype=np.float32)
    for y in range(img_height):
      for x in range(img_width):
        # pdb.set_trace()
        # u_vec = self.calc_dir_vec([y, x], img_height, img_width)
        u_vec = self.pos_encoding_map[y,x,:]
        peak_dir_encoding_map[y,x,0] = np.exp(100.0 * (np.dot(u_vec, peak_dir_vec) - 1.0))

    return peak_dir_encoding_map
  
  def calc_peak_intensity_encoding(self, peak_dir_encoding_map, peak_intensity_vec, img_height, img_width):
    peak_intensity_encoding_map = np.zeros((img_height, img_width, 3), dtype=np.float32)
    for y in range(img_height):
      for x in range(img_width):
        # pdb.set_trace()
        # u_vec = self.calc_dir_vec([y, x], img_height, img_width)
        if peak_dir_encoding_map[y,x,0] >= 0.98:
          peak_intensity_encoding_map[y,x,0] = peak_intensity_vec[0]
          peak_intensity_encoding_map[y,x,1] = peak_intensity_vec[1]
          peak_intensity_encoding_map[y,x,2] = peak_intensity_vec[2]
    
    return peak_intensity_encoding_map

  def __len__(self):
    return len(self.annotations)

  def __getitem__(self, index):
    hdr_img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0] + '.hdr')
    ldr_img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1] + '.hdr')
    # pdb.set_trace()
    azimuth = int(self.annotations.iloc[index, 2])
    elevation = int(self.annotations.iloc[index, 3])
    sun_pose = self.vMF(azimuth ,elevation, IMSHAPE[0], IMSHAPE[1])
    # cv2.imwrite(os.path.join('/mnt/ssd/home/chenbai/sunpose_pytorch_test', self.annotations.iloc[index, 0] + '_sunpose_test.hdr'), 1000.0 * sun_pose.reshape(32,128))
    # pdb.set_trace()
    # read ldr image
    ldr_img = cv2.imread(ldr_img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    # cv2.imwrite(os.path.join('/mnt/ssd/home/chenbai/sunpose_pytorch_test', self.annotations.iloc[index, 1] + '.hdr'), ldr_img)
    # ldr_img = cv2.cvtColor(ldr_img, cv2.COLOR_BGR2RGB)
    # img_height, img_width, img_ch = ldr_img.shape
    try:
      img_height, img_width, img_ch = ldr_img.shape
    except KeyboardInterrupt:
      return
    except Exception as e:
      print(e)
      pdb.set_trace()
    # # print('LDR before')
    # # print(ldr_img)
    if self.ldr_transform:
      ldr_img = self.ldr_transform(ldr_img)
    # # print('LDR after')
    # # print(ldr_img)

    
    # read hdr image
    # hdr_img = cv2.imread(hdr_img_path, flags=cv2.IMREAD_ANYDEPTH)
    hdr_img = cv2.imread(hdr_img_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
    # print('HDR max val: ', hdr_img.max())
    # hdr_img = cv2.cvtColor(hdr_img, cv2.COLOR_BGR2RGB)
    hdr_img = np.clip(hdr_img, 0, self.hdr_clip_threshold)
    # # hdr_img = hdr_img / self.hdr_clip_threshold
    # ## Normalize hdr to scale [0, 1] as a 3-channel probability map
    # # print('HDR shape,',hdr_img.shape)
    # # print('cv2 HDR img before')
    # # print(hdr_img.dtype)
    # # print(hdr_img.shape)
    # # print(hdr_img[335][1228][0])
    if self.hdr_transform:
      hdr_img = self.hdr_transform(hdr_img)
    # # print('cv2 HDR img after')
    # # print(hdr_img.dtype)
    # # print(hdr_img.shape)
    # # print(hdr_img[0][335][1228])
    # # pdb.set_trace()
    return (hdr_img, ldr_img, sun_pose)

if __name__ == "__main__":
  print('Custom dataset main')

  cvs_file_dir = '/media/sandisk4T_2/lavel_sky_dataset/laval_sky_cut_processed_tf/dataset_128_32/train/train_refine.csv'
  root_dir = '/media/sandisk4T_2/lavel_sky_dataset/laval_sky_cut_processed_tf/dataset_128_32/train/hdr/'
  num_train_sample = 16000
  num_test_sample = 700

  ## copy printed normalize paramter from "CalcNormParamters" to transform
  ldr_transform = transforms.Compose([
      transforms.ToPILImage(),
      transforms.RandomHorizontalFlip(p=0.25),
      transforms.ToTensor(),
      transforms.Normalize([0.6015054, 0.6635834, 0.6955328], [0.2799979, 0.26795584, 0.28252074])
  ])
  hdr_transform = transforms.Compose([
      transforms.ToTensor()
  ])
  peak_intensity_transform = transforms.Compose([
      transforms.Normalize([51733.773, 50709.19, 37421.99], [76532.97, 70329.04, 59088.164])
  ])
  peak_dir_transform = transforms.Compose([
      transforms.Normalize([0.19018753, -0.29801416, 0.24725443], [0.4686135, 0.57455647, 0.5155442])
  ])

  # hdr_dataset = HDRDataset('/home/chengzhang/projects/HDR_cut_dataset/dataset_generate.csv','/home/chengzhang/projects/HDR_cut_dataset/',transform)
  hdr_dataset = HDRDataset(cvs_file_dir, root_dir, ldr_transform=ldr_transform,\
                           hdr_transform=hdr_transform, peak_intensity_transform=peak_intensity_transform, peak_dir_transform=peak_dir_transform)
  train_set, test_set = torch.utils.data.random_split(hdr_dataset, [num_train_sample,num_test_sample]) # split data number [train, test]
  train_loader = DataLoader(dataset = train_set, batch_size = 32, shuffle = True)
  test_loader = DataLoader(dataset = test_set, batch_size = 1, shuffle = True)

  for batch_idx, (hdr_imgs, ldr_imgs, sun_poses) in enumerate(train_loader):
    # print('LDR img')
    # print(ldr_imgs)
    # print('peak intensity')
    # print(peak_intensities)
    # print('peak dir')
    # print(peak_dirs)
    print('HDR img')
    print(hdr_imgs.shape)
