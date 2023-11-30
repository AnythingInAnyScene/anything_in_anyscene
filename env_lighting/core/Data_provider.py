import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import pdb
import random

class VDS_half_dataset(Dataset):
    def __init__(self, root:str, training:bool, increase:bool, img_num=7, img_size=(256,256), center=0, transform=None, cache=False, augment=False, norm=False, VDS_our=False) -> None:
        super().__init__()
        self.root = root
        self.img_num = (img_num // 2) +1 # 4, 應該是分變亮變暗, 分別從 0到 -1~-3, 0到 1~3 都是4張
        self.center = center
        self.cache = cache
        self.augment = augment

        if training:
            if VDS_our == False:
                self.root += '/train/set'
            else:
                self.root += '/train_our/set'
                print("Using our VDS training set")
        else:
            if VDS_our == False:
                self.root += '/test/set'
            else:
                self.root += '/train_our/set'
                print("Using our VDS testing set")

        if increase:
            self.base = 0
        else:
            self.base = -1 * self.img_num +1 
            # -3 (decrease), 如果是decrease, 那等等的img_ind(Tar_exp) 加上這個就會是負值(變暗part)

        self.data_stack = os.listdir(self.root) # ['t29', 't49', ....'t30']

        if transform:
            self.transform = transform
        else:
            if norm:
                self.transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(img_size),
                    transforms.ToTensor()
                ])



        if self.cache == True:
            #-----------------------cache-----------------------------
            print("Cache img start!!")
            self.img_cache = {}
            for scene_num in self.data_stack:
                print("cache scene ", scene_num, " image...")
                scene_dict = {}
                access_path = self.root + "/" + scene_num + "/"

                LDR_neg3_name = scene_num + "_-3EV_true.jpg.png"
                LDR_neg2_name = scene_num + "_-2EV_true.jpg.png"
                LDR_neg1_name = scene_num + "_-1EV_true.jpg.png"
                LDR_0_name = scene_num + "_0EV_true.jpg.png"
                LDR_pos1_name = scene_num + "_1EV_true.jpg.png"
                LDR_pos2_name = scene_num + "_2EV_true.jpg.png"
                LDR_pos3_name = scene_num + "_3EV_true.jpg.png"

                LDR_neg3 = self.transform(Image.open(access_path + LDR_neg3_name).convert('RGB'))
                LDR_neg2 = self.transform(Image.open(access_path + LDR_neg2_name).convert('RGB'))
                LDR_neg1 = self.transform(Image.open(access_path + LDR_neg1_name).convert('RGB'))
                LDR_0 = self.transform(Image.open(access_path + LDR_0_name).convert('RGB'))
                LDR_pos1 = self.transform(Image.open(access_path + LDR_pos1_name).convert('RGB'))
                LDR_pos2 = self.transform(Image.open(access_path + LDR_pos2_name).convert('RGB'))
                LDR_pos3 = self.transform(Image.open(access_path + LDR_pos3_name).convert('RGB'))

                #pdb.set_trace()

                scene_dict["-3"] = LDR_neg3
                scene_dict["-2"] = LDR_neg2
                scene_dict["-1"] = LDR_neg1
                scene_dict["0"] = LDR_0
                scene_dict["1"] = LDR_pos1
                scene_dict["2"] = LDR_pos2
                scene_dict["3"] = LDR_pos3

                self.img_cache[scene_num] = scene_dict
            print("Cache img finish!!")
            # --------------------------------------------------------           



    def stack_img(self, stack, EV):
        # format: ./train_set/t1/t1_0EV_true.jpg
        path = self.root + '/' + stack
        path = path + '/' + stack + '_' + str(EV) + 'EV_true.jpg.png'
        return path

    def __len__(self) -> int:
        return len(self.data_stack) * self.img_num

    def __getitem__(self, index):
        
        """
        half的話, self.img_num從設定的7變4.
        index 這輪random num = 158
        """
        stack_ind = index // self.img_num  #39 => 用來sample random scene (t10?, t14?...)
        set_ind = self.center #0 => 輸入的source EV值
        img_ind = (index % self.img_num) + self.base #2 => 目標的image target EV

        if self.cache == False: 
            img_path = self.stack_img(self.data_stack[stack_ind], set_ind)
            #'/home/skchen/ML_practice/LIIF_on_HDR/VDS_dataset/train/set/t92/t92_0EV_true.jpg.png'

            img = Image.open(img_path)
            if self.transform:
                img = self.transform(img)
            
            gt_img_path = self.stack_img(self.data_stack[stack_ind], set_ind + img_ind)
            #'/home/skchen/ML_practice/LIIF_on_HDR/VDS_dataset/train/set/t92/t92_2EV_true.jpg.png'

            gt_img = Image.open(gt_img_path)
            if self.transform:
                gt_img = self.transform(gt_img)
                
            step = torch.tensor([set_ind + img_ind], dtype=torch.float32)
            origin = torch.tensor([set_ind], dtype=torch.float32)
            #print(stack_ind, img_ind, set_ind)

            #Augmentation
            if self.augment:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                dflip = random.random() < 0.5

                #print("hflip: ", hflip, ", vflip: ", vflip, ", dflip: ", dflip)

                def augment(x):
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    if dflip:
                        x = x.transpose(-2, -1)
                    return x

                gt_img = augment(gt_img)
                img = augment(img)
            
            return img, gt_img, step, origin

        elif self.cache == True:
            scene = self.data_stack[stack_ind]
            source_ev = str(set_ind)
            target_ev = str(img_ind)

            img = self.img_cache[scene][source_ev]
            gt_img = self.img_cache[scene][target_ev]
            step = torch.tensor([set_ind + img_ind], dtype=torch.float32)
            origin = torch.tensor([set_ind], dtype=torch.float32)

            #Augmentation
            if self.augment:
                hflip = random.random() < 0.5
                vflip = random.random() < 0.5
                dflip = random.random() < 0.5

                #print("hflip: ", hflip, ", vflip: ", vflip, ", dflip: ", dflip)

                def augment(x):
                    if hflip:
                        x = x.flip(-2)
                    if vflip:
                        x = x.flip(-1)
                    if dflip:
                        x = x.transpose(-2, -1)
                    return x

                gt_img = augment(gt_img)
                img = augment(img)

            return img, gt_img, step, origin


def build_dataset(args):
    """Builds the dataset."""
    implemented_datasets = ('half')
    assert args.set_name in implemented_datasets

    one_side_datasets = ('half', 'half_full', 'eye')
    one_side = False
    if args.set_name in one_side_datasets:
        implemented_mode = ('increase', 'decrease')
        assert args.dataset_mode in implemented_mode
        one_side = True

        if args.dataset_mode == 'increase':
            increase = True
        else:
            increase = False

    dataset = None
    test_set = None
    assert args.img_height % 64 == 0
    img_size = (args.img_height, args.img_height)

    if args.set_name == 'half':
        dataset = VDS_half_dataset(args.data_root, training=True, increase=increase, img_num=args.img_num, img_size=img_size, cache=args.cache, augment=args.augment, norm=args.norm, VDS_our=args.VDS_our)
        test_set = VDS_half_dataset(args.data_root, training=False, increase=increase, img_num=args.img_num, img_size=img_size, cache=args.cache, augment=False, norm=args.norm, VDS_our=args.VDS_our)

    if one_side:
        print('mode:', args.dataset_mode,'img_num:', args.img_num, 'img_set:', args.img_set, 'img_size:', img_size)
    else:
        print('img_num:', args.img_num, 'img_set:', args.img_set, 'img_size:', img_size)

    return dataset, test_set

def build_eval_dataset(args):
    """Builds the dataset."""
    implemented_datasets = ('base', 'full', 'half', 'half_full', 'eye')
    assert args.set_name in implemented_datasets

    one_side_datasets = ('half', 'half_full', 'eye')
    one_side = False
    if args.set_name in one_side_datasets:
        implemented_mode = ('increase', 'decrease')
        assert args.dataset_mode in implemented_mode
        one_side = True

        if args.dataset_mode == 'increase':
            increase = True
        else:
            increase = False

    dataset = None
    test_set = None
    assert args.img_height % 64 == 0
    img_size = (args.img_height, args.img_height)

    if args.set_name == 'base' or args.set_name == 'full':
        dataset = VDS_eval_dataset(args.data_root, training=True, img_num=args.img_num, img_size=img_size)
        test_set = VDS_eval_dataset(args.data_root, training=False, img_num=args.img_num, img_size=img_size)

    if args.set_name == 'half' or args.set_name == 'half_full':
        dataset = VDS_half_dataset(args.data_root, training=True, increase=increase, img_num=args.img_num, img_size=img_size)
        test_set = VDS_half_dataset(args.data_root, training=False, increase=increase, img_num=args.img_num, img_size=img_size)

    if args.set_name == 'eye':
        dataset = None
        test_set = Eye_semi_dataset(args.data_root, increase=increase, img_num=args.img_num, img_size=img_size)

    if one_side:
        print('mode:', args.dataset_mode,'img_num:', args.img_num, 'img_set:', args.img_set, 'img_size:', img_size)
    else:
        print('img_num:', args.img_num, 'img_set:', args.img_set, 'img_size:', img_size)

    return dataset, test_set
