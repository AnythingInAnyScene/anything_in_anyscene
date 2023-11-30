import os
import sys
import cv2
import numpy as np
import anything_in_anyscene.env_lighting.lib.Perspec2Equirec as P2E


class Perspective:
    def __init__(self, img_array , F_T_P_array ):
        
        assert len(img_array)==len(F_T_P_array)
        
        self.img_array = img_array
        self.F_T_P_array = F_T_P_array
    

    def GetEquirec(self,height,width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #
        merge_image = np.zeros((height,width,3), dtype=np.float32)
        merge_mask = np.zeros((height,width,3), dtype=np.float32)

        for img_dir,[F,T,P] in zip (self.img_array,self.F_T_P_array):
            per = P2E.Perspective(img_dir,F,T,P)        # Load equirectangular image
            img, mask = per.GetEquirec(height,width)   # Specify parameters(FOV, theta, phi, height, width)
            # merge_image += img

            merge_image[merge_image<=0] = img[merge_image<=0]
            merge_mask += mask

        merge_mask = np.where(merge_mask==0,1,merge_mask)
        # merge_image = (np.divide(merge_image,merge_mask))

        
        return merge_image
        






