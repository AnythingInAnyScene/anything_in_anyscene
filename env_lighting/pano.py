import cv2
import numpy as np
import os, glob
import json
import matplotlib.pyplot as plt
import math
import imageio
from argparse import ArgumentParser

import anything_in_anyscene.env_lighting.lib.Equirec2Perspec as E2P
import anything_in_anyscene.env_lighting.lib.Perspec2Equirec as P2E
import anything_in_anyscene.env_lighting.lib.multi_Perspec2Equirec as m_P2E


def quaternion_to_euler(w, x, y, z):
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))

    # Pitch (attitude) angle
    pitch = math.asin(2 * (w * y - x * z))

    # Roll (bank) angle
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))

    # Convert angles from radians to degrees if needed
    yaw = math.degrees(yaw)
    pitch = math.degrees(pitch)
    roll = math.degrees(roll)

    return yaw, pitch, roll

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path")
    parser.add_argument("--hdr_path")
    args = parser.parse_args()

    image_path = os.path.join(args.input_path, "..")
    folder_name = args.input_path.split("/")[-1]
    hdr_path = args.hdr_path
    output_hdr_path = os.path.join(args.input_path, "hdr_results", "output_pano.hdr")
    output_hdr_tonemap_path = os.path.join(args.input_path, "hdr_results", "output_pano.png")
    output_original_path = os.path.join(args.input_path, "hdr_results", "original_pano.png")


    cam_input_list = []
    cam_hdr_list = []
    cam_fov_list = []
    camera_list = ["front_left_camera", "left_camera", "front_right_camera", "right_camera", "back_camera"]

    for cam_id in camera_list:
        cam_hdr_list.append(cv2.imread(os.path.join(args.hdr_path, cam_id, "00.hdr")))
        cam_input_list.append(cv2.imread(os.path.join(args.input_path, cam_id, "00.jpg")))
        with open(os.path.join(args.input_path, cam_id, "poses.json"), "r") as file:
            json_data = file.read()
        extrinsic_data = json.loads(json_data)
        w = extrinsic_data[0]['heading']['w']
        x = extrinsic_data[0]['heading']['x']
        y = extrinsic_data[0]['heading']['y']
        z = extrinsic_data[0]['heading']['z']
        yaw, pitch, roll = quaternion_to_euler(w, x, y, z)
        yaw = -yaw-45
        if cam_id == "front_camera":
            cam_fov_list.append([50, yaw, pitch])
        else:
            cam_fov_list.append([107, yaw, pitch])
        print(cam_id, yaw, pitch)


    pano_height, pano_width = 512, 1024
    pano = np.zeros([pano_height, pano_width])

    equ = m_P2E.Perspective(cam_hdr_list, cam_fov_list)    
    pano_hdr = equ.GetEquirec(pano_height, pano_width) 
    gamma = 1
    pano_hdr_tonemap = (pano_hdr / 1).astype('float') ** (1 / gamma)       
    pano_hdr_tonemap = (pano_hdr_tonemap * 255).clip(0, 255).astype('uint8')
    equ = m_P2E.Perspective(cam_input_list, cam_fov_list)    
    pano_original = equ.GetEquirec(pano_height, pano_width)
        
    print("hdr max_value", pano_hdr.max())
    pano_hdr = imageio.core.util.Array(pano_hdr[:,:,2::-1])
    imageio.imsave(output_hdr_path, pano_hdr)
    cv2.imwrite(output_hdr_tonemap_path, pano_hdr_tonemap)
    cv2.imwrite(output_original_path, pano_original)



    
    