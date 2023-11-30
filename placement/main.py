from anything_in_anyscene.placement.models.YoloP import get_yolop
from anything_in_anyscene.placement.models.PackNet import get_packnet
from anything_in_anyscene.placement.models.PackNet.utils import model_utils
import anything_in_anyscene.placement.utils as utils
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
import tqdm
import torch
import yaml
import pickle
import argparse
import os
import pkg_resources
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Order GPUs by PCI bus ID


def select_device(device='', batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        # os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(
        ), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (
                batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = f'Using torch {torch.__version__} '
        print("%sCUDA:%s (%s, %dMB)" %
              (s, device, x[int(device)].name, x[int(device)].total_memory / c))
    else:
        print(f'Using torch {torch.__version__} CPU')

    print('')  # skip a line
    return torch.device('cuda:'+str(device) if cuda else 'cpu')


def infer_depth(cfg, model, img_list, device):
    trans = transforms.Compose([
        #   transforms.Resize((128, 256))       # must be 64x
        transforms.Resize(
            cfg.img_resize, interpolation=Image.Resampling.LANCZOS),
        # range [0,255] to [0.0, 1.0], np.uint8 -> torch.float23, img = img/255
        transforms.ToTensor(),
    ])

    test_img = Image.open(img_list[0])
    W_, H_ = test_img.size
    restore_trans = transforms.Resize(
        (H_, W_), transforms.InterpolationMode.NEAREST)

    model.to(device)
    model.eval()
    # test_run once
    test_img = trans(test_img)
    test_img = torch.unsqueeze(test_img, dim=0)
    test_img = test_img.to(device)
    pred_inv_depth = model.depth_net(test_img)['inv_depths'][0]
    pred_depth = model_utils.inv2depth(pred_inv_depth)
    pred_depth = restore_trans(pred_depth)
    pred_depth = pred_depth.detach().squeeze().cpu()
    print("Test RUN: smallest_range: {:.2f}, largest range: {:.2f}".format(
        pred_depth.min(), pred_depth.max()))

    # run all
    for img_path in tqdm.tqdm(img_list):
        # opencv
        # ori_img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        # img_ = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        # img_ = trans(img_)
        ori_img = Image.open(img_path)
        img_ = trans(ori_img)
        img_ = torch.unsqueeze(img_, dim=0)
        img_ = img_.to(device)

        pred_inv_depth = model.depth_net(img_)['inv_depths'][0]
        pred_depth = model_utils.inv2depth(pred_inv_depth)
        pred_depth = restore_trans(pred_depth)
        pred_depth = pred_depth.detach().squeeze().cpu()

        if cfg.verbose:
            print("smallest_range: {:.2f}, largest range: {:.2f}".format(
                pred_depth.min(), pred_depth.max()))

        save_depth_name = img_path.split("/")[-1].split(".")[0] + "_depth.npy"
        np.save(os.path.join(cfg.depth_dir, save_depth_name), pred_depth)

        if cfg.debug_img:
            if not os.path.exists(cfg.depth_debug_dir):
                os.makedirs(cfg.depth_debug_dir)
            # pred_depth_color = transforms.ToPILImage()((pred_depth * 256).int())
            pred_depth_color = utils.convert_color(pred_depth)
            # check_img = cv2.imread(img_path, flags=cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            # check_img = cv2.vconcat([check_img, pred_depth_color])
            check_img = cv2.vconcat(
                [utils.pil2opencv(ori_img), pred_depth_color])
            save_img_name = img_path.split(
                "/")[-1].split(".")[0] + "_depth.jpg"
            cv2.imwrite(os.path.join(
                cfg.depth_debug_dir, save_img_name), check_img)


def infer_seg(cfg, model, img_list, device):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    model.to(device)
    model.eval()

    # test_run once
    test_img = cv2.imread(
        img_list[0], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
    H_, W_ = test_img.shape[:2]
    test_img_resize, _, (pad_w, pad_h) = utils.letterbox_for_img(
        test_img, new_shape=cfg.img_resize, auto=True)
    pad_w = int(pad_w)
    pad_h = int(pad_h)
    # ratio = (H_ / cfg.img_resize[0], W_ / cfg.img_resize[1])
    test_img_resize = np.ascontiguousarray(test_img_resize)

    test_img_resize = transform(test_img_resize).to(device)
    test_img_resize = torch.unsqueeze(test_img_resize, dim=0)

    _ = model(test_img_resize)
    print("Test RUN: Pass")

    # run all
    for img_path in tqdm.tqdm(img_list):
        ori_img = cv2.imread(img_path, cv2.IMREAD_COLOR |
                             cv2.IMREAD_IGNORE_ORIENTATION)  # BGR
        img_resize, _, _ = utils.letterbox_for_img(
            ori_img, new_shape=cfg.img_resize, auto=True)
        img_resize = np.ascontiguousarray(img_resize)
        img_resize = transform(img_resize).to(device)
        img_resize = torch.unsqueeze(img_resize, dim=0)

        _, da_seg_out, ll_seg_out = model(img_resize)

        da_predict = da_seg_out[:, :, pad_h:(
            cfg.img_resize[0]-pad_h), pad_w:(cfg.img_resize[1]-pad_w)]
        da_seg_mask = torch.nn.functional.interpolate(
            da_predict, size=(H_, W_), mode='bilinear')
        _, da_seg_mask = torch.max(da_seg_mask, 1)
        da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        # da_seg_mask = morphological_process(da_seg_mask, kernel_size=7)

        ll_predict = ll_seg_out[:, :, pad_h:(
            cfg.img_resize[0]-pad_h), pad_w:(cfg.img_resize[1]-pad_w)]
        ll_seg_mask = torch.nn.functional.interpolate(
            ll_predict, size=(H_, W_), mode='bilinear')
        _, ll_seg_mask = torch.max(ll_seg_mask, 1)
        ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()

        img_det, img_seg = utils.show_seg_result(
            ori_img, (da_seg_mask, ll_seg_mask), cfg.ground_mask, cfg.lane_mask)
        save_seg_name = img_path.split("/")[-1].split(".")[0] + "_seg.png"
        seg_save_path = os.path.join(cfg.seg_dir, save_seg_name)
        cv2.imwrite(seg_save_path, img_seg)

        if cfg.debug_img:
            if not os.path.exists(cfg.seg_debug_dir):
                os.makedirs(cfg.seg_debug_dir)
            save_img_name = img_path.split("/")[-1]
            cv2.imwrite(os.path.join(
                cfg.seg_debug_dir, save_img_name), img_det)


def main(args):
    # ----------------------------------------------------------------------
    #  set parameters
    # ----------------------------------------------------------------------
    if not args.cfg_file is None:
        try:
            with open(args.cfg_file, 'r') as f:
                yml_args = yaml.load(f, Loader=yaml.FullLoader)
        except:
            yaml_data = pkg_resources.resource_string("anything_in_anyscene.placement", args.cfg_file)
            yml_args = yaml.safe_load(yaml_data)
        cfg = utils.merge_cfg(vars(args), yml_args)
        cfg = argparse.Namespace(**cfg)
    else:
        cfg = args

    # if cfg.img_resize is None:
    #     cfg.img_resize = yml_args['img_resize']
    if not os.path.exists(cfg.save_dir):
        os.makedirs(cfg.save_dir)

    if not cfg.depth_dir:
        cfg.depth_dir = os.path.join(cfg.save_dir, cfg.depth_subdir)
    if not os.path.exists(cfg.depth_dir):
        os.makedirs(cfg.depth_dir)
    if not cfg.seg_dir:
        cfg.seg_dir = os.path.join(cfg.save_dir, cfg.seg_subdir)
    if not os.path.exists(cfg.seg_dir):
        os.makedirs(cfg.seg_dir)
    cfg.place_dir = os.path.join(cfg.save_dir, cfg.place_subdir)
    if not os.path.exists(cfg.place_dir):
        os.makedirs(cfg.place_dir)
    # cfg.__setattr__('depth_dir', os.path.join(cfg.save_dir, "depth"))
    # cfg.__setattr__('seg_dir', os.path.join(cfg.save_dir, "seg"))
    cfg.__setattr__('depth_debug_dir', os.path.join(cfg.debug_dir, "depth"))
    cfg.__setattr__('seg_debug_dir', os.path.join(cfg.debug_dir, "seg"))
    cfg.__setattr__('rgb_debug_dir', os.path.join(cfg.debug_dir, "rgb"))
    # infer from yaml
    cfg.__setattr__('cam_extrinsic', None)
    cfg.__setattr__('cam_intrinsic', None)
    cfg.__setattr__('Tcam0_2_lidar', None)
    cfg.__setattr__('mesh_info', None)
    cfg.__setattr__('label_valid', False)
    cfg.__setattr__('lidar_valid', True)

    # src_dir:xxx/xxx/[clip_key]/cam0
    clip_key = cfg.src_dir.split('/')[-2]
    img_list = utils.ls_folder_img(cfg.src_dir)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< camera parameters >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    calibration_json = utils.read_calibration(cfg.calibration)
    cfg.cam_intrinsic_dict, cfg.cam_intrinsic = utils.read_intrinsic(calibration_json, clip_key)
    # T_w2cam, list of 4x4
    cfg.cam_extrinsic = utils.read_extrinsic(calibration_json, clip_key)
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< lidar check >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if os.path.exists(cfg.lidar_dir):
        lidars = utils.ls_folder_img(cfg.lidar_dir, img_formats=".json")
        lack_lidars = utils.infer_img_filter(img_list, lidars, suffix_match='')
        if len(lack_lidars) <= 0:
            cfg.lidar_valid = True
        else:
            utils.print_yellow("lack lidar {} files !".format(len(lack_lidars)))
    if cfg.debug_3d or cfg.lidar_valid:
        # incase no "lidar2" in calibration
        try:
            T_w2lidar = utils.read_extrinsic(calibration_json, clip_key, "lidar2")
            Tcam0_2_lidar = utils.get_trans(cfg.cam_extrinsic, T_w2lidar)
            cfg.Tcam0_2_lidar = Tcam0_2_lidar
        except KeyError:
            cfg.lidar_valid = False
            utils.print_red("no availabel calibration_json on key ['lidar2']")
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< label check >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if os.path.exists(cfg.label_dir):
        labels = utils.ls_folder_img(cfg.label_dir, img_formats=".json")
        lack_labels = utils.infer_img_filter(img_list, labels, suffix_match='')
        if len(lack_labels) <= 0:
            cfg.label_valid = True
        else:
            utils.print_yellow("lack label {} files !".format(len(lack_labels)))
    if not cfg.label_valid:
        utils.print_yellow("labels not valid in: \n{}".format(cfg.label_dir))
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<< mesh check >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    if os.path.exists(cfg.mesh_file):
        cfg.mesh_info = utils.load_mesh_list(cfg.mesh_file)
    else:
        utils.print_yellow("lack of mesh info {}, use default bbox setting.".format(cfg.mesh_file))
    print("====="*20)
    utils.print_green("using settings: \n" +
                      "".join(["-{}: {}\n".format(k, v) for k, v in vars(cfg).items()]))
    print("====="*20)

    # ----------------------------------------------------------------------
    #  prepare input
    # ----------------------------------------------------------------------
    print("====="*20)
    utils.print_red("0. sfm file check ...")
    with open(cfg.track_file, 'rb') as fp:
        track_file = pickle.load(fp)
        utils.print_green(
            "Pose file contains {} frames!".format(len(track_file)))
        track_file = utils.sfm_filter(track_file, match_img_list=img_list)
        utils.print_green("\t {} frames valid!".format(len(track_file)))
    # ! save idx, img name
    file_idx_img_name = os.path.join(cfg.save_dir, "map_idx_img.txt")
    with open(file_idx_img_name, 'w+') as fp:
        for idx, pdata in enumerate(track_file):
            img_name = pdata[0]
            fp.write(f"{str(idx)} {img_name}\n")
    assert len(track_file) > cfg.min_tracknum, utils.print_red(
        "no enough valid sfm frames: {}".format(len(track_file)))
    print("====="*20)

    # ----------------------------------------------------------------------
    #  prepare model / mask / depth
    # ----------------------------------------------------------------------
    seg_list = utils.ls_folder_img(cfg.seg_dir, img_formats=['.png'])
    seg_list = utils.infer_img_filter(img_list, seg_list, '_seg')
    device = select_device(cfg.device)
    print("====="*20)
    if seg_list:
        yolop = get_yolop(cfg.yolop_ckp)
        utils.print_red("1.1 prepare segmentation...")
        infer_seg(cfg, yolop, seg_list, device)
        del yolop
    else:
        utils.print_red(
            "1.1 skip segmentation because {} already has...".format(cfg.seg_dir))
    print("====="*20)

    # no need depth
    # depth_list = utils.ls_folder_img(cfg.depth_dir, img_formats=['.npy'])
    # depth_list = utils.infer_img_filter(img_list, depth_list, '_depth')
    # print("====="*20)
    # if depth_list:
    #     packnet = get_packnet(cfg.packnet_ckp)
    #     utils.print_red("1.2 prepare depth...")
    #     infer_depth(cfg, packnet, depth_list, device)
    #     del packnet
    # else:
    #     utils.print_red(
    #         "1.1 skip depth because {} already has...".format(cfg.depth_dir))
    # print("====="*20)

    # ----------------------------------------------------------------------
    #  select point, auto / manual
    # ----------------------------------------------------------------------
    print("====="*20)
    utils.print_red("2. select point...")
    print("====="*20)
    res = {}
    if cfg.manual is None:
        utils.print_green("auto place point selection.")
        res = utils.auto_mesh_place(cfg, track_file)

    else:
        cfg.check_collision = False  # ensure false if manual
        rgb_path, d_path, seg_path = utils.track_file2rgbdseg_path(track_file, cfg.ref_idx,
                                                                   cfg.src_dir, cfg.depth_dir, cfg.seg_dir)
        utils.print_yellow("select at img {}".format(rgb_path))
        _, d, _ = utils.read_rgb_d_seg(rgb_path, d_path, seg_path)
        if len(cfg.manual) >= 2 and \
                cfg.manual[0] >= 0 and cfg.manual[0] <= cfg.cam_intrinsic_dict['width'] and \
                cfg.manual[1] >= 0 and cfg.manual[1] <= cfg.cam_intrinsic_dict['height']:
            utils.print_red(
                "manual place point at ({}, {}) in pixel (u,v)".format(*cfg.manual[:2]))
            a_place_point = utils.uvd2xyz(
                cfg.manual[0], cfg.manual[1], d[cfg.manual[1], cfg.manual[0]], cfg.cam_intrinsic)
            a_place_point = a_place_point.reshape(1, 3)
            res['0'] = {"place_result_3d": a_place_point.copy(),
                                         "begin_idx": cfg.ref_idx,
                                         "end_idx": len(track_file),
                                         "valid_frame_count": len(track_file) - cfg.ref_idx
                                         }
        else:
            utils.print_red(
                "not a valid place point at ({}, {}) in pixel (u,v)".format(*cfg.manual[:2]))

    utils.print_yellow(f"Get {len(res)} place points.")
    if res is None or len(res) <= 0:
        utils.print_red("Error: no valid place point!")
        exit(-1)

    # # ----------------------------------------------------------------------
    # #  curvature check
    # # ----------------------------------------------------------------------
    # if cfg.label_valid and not cfg.fix_pose:
    #     for p_idx, p_info in res.items():
    #         res[p_idx]['curvature_valid'] = True
    #         # find lld
    #         begin_idx = res[p_idx]['begin_idx']
    #         label_file = utils.track_file2label_path(
    #             track_file if p_info["new_pose"] is None else p_info["new_pose"],
    #             begin_idx, cfg.label_dir)
    #         ego_lanes = utils.lld_get_ego_lane(label_file)
    #         if ego_lanes:
    #             ego_lanes = [np.array(ego_lane).reshape(len(ego_lane), 2) for ego_lane in ego_lanes]
    #             all_path = p_info["obj_trajectory_2d"].copy()
    #             res[p_idx]['curvature_valid'] = utils.curvature_filter(ego_lanes, all_path, cfg.curvature_diff)
    # else:
    #     # no label or use optical flow
    #     utils.print_yellow(f"no curvature check!")
    
    # ----------------------------------------------------------------------
    #  debug
    # ----------------------------------------------------------------------
    if cfg.debug_img or cfg.debug_3d:
        utils.debug_all(cfg, track_file, res)

    # ----------------------------------------------------------------------
    #  Save results
    # ----------------------------------------------------------------------

    print("====="*20)
    utils.print_red("4. save results...")
    print("====="*20)
    for p_idx, p_info in res.items():
        if p_info['valid_frame_count'] > 0:
            if (not "curvature_valid" in p_info) or p_info['curvature_valid']:
                utils.print_yellow(f"Saving point {p_idx}...")
                save_path = os.path.join(cfg.place_dir, f"point{p_idx}")
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # np.savetxt(os.path.join(save_path, "xyz.txt"),
                #         p_info['place_result_3d'].reshape(1, 3))
                with open(os.path.join(save_path, "xyz.txt"), "w+") as f:
                    f.write(p_info['use_mesh'] + '\n' if p_info['use_mesh'] else 'None\n')
                    xyz = p_info['place_result_3d'].flatten().tolist()
                    f.write(" ".join([str(i) for i in xyz]) + '\n')
                    f.write(" ".join([str(i) for i in p_info['init_pose']]) + '\n')
                np.save(os.path.join(save_path, "plane_R_Select2Ref.npy"),
                        p_info['ground_plane'])
                np.savetxt(os.path.join(save_path, "valid_frame.txt"), [
                        [p_info['begin_idx'], p_info['end_idx']], [p_info['select_idx'], p_info['valid_frame_count']]], fmt="%d")
                if not p_info["new_pose"] is None:
                    new_track_file = os.path.join(save_path, "images_fixby_optical.txt")
                    utils.print_yellow(f"save new pose file to {new_track_file}")
                    with open(new_track_file, 'wb') as fp:
                        pickle.dump(p_info["new_pose"], fp)
            else:
                utils.print_yellow("[curvature check]: not a valid traj")
        else:
            continue
    print("All Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ================================================== general args ==================================================
    # input
    parser.add_argument('--cfg-file', type=str,
                        default='./config.yaml', help='config file')
    parser.add_argument('--track-file', type=str, help='trajectory file')
    parser.add_argument('--mesh-file', type=str, help='models_list.txt file')
    parser.add_argument('--calibration', type=str,
                        help='calibration.json file')
    # file/folder   ex:inference/images
    parser.add_argument('--src-dir', type=str, help='source')
    parser.add_argument('--lidar-dir', type=str, help='lidar source')
    parser.add_argument('--label-dir', type=str, help='label source')
    # output
    parser.add_argument('--save-dir', type=str,
                        help='directory to save results')
    parser.add_argument('--depth-dir', type=str,
                        help='directory to save depth results')
    parser.add_argument('--seg-dir', type=str,
                        help='directory to save seg results')
    # parser.add_argument('--place-subdir', type=str, help='directory to save place points results under save-dir')
    # debug
    parser.add_argument('--debug-dir', type=str,
                        help='directory to save debug results')
    parser.add_argument('--debug-img', action='store_true',
                        help='save debug result images')
    parser.add_argument('--debug-3d', action='store_true',
                        help='save debug 3d points')

    parser.add_argument('--verbose', default=False,
                        action='store_true', help='show more informations')
    parser.add_argument('--fix-pose', default=False,
                        action='store_true', help='show more informations')
    parser.add_argument('--device', type=str,
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--img-resize', type=int, nargs='+',
                        default=None, help='infer img size in packnet/yolop')
    # ================================================== yolop args ==================================================
    parser.add_argument('--yolop-ckp', type=str, help='model.pth path(s)')
    # ================================================== packet args ==================================================
    parser.add_argument('--packnet-ckp', type=str, help='Checkpoint (.ckpt)')
    # ================================================== place point args ==================================================
    parser.add_argument('--ref-idx', type=int,
                        help='the 1st frame in track-file to place point')
    parser.add_argument('--roll-back', action='store_true',
                        help='use the last point, back to 1st frame')
    parser.add_argument('--check-collision', action='store_true',
                        help='check collision when select points')
    parser.add_argument('--manual', type=int, nargs='+',
                        help='place point in pixel(u, v) by manual')
    args = parser.parse_args()
    main(args)
