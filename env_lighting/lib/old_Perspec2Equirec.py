import os
import sys
import cv2
import numpy as np

class Perspective:
    def __init__(self, img_name , FOV, THETA, PHI ):
        self._img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        [self._height, self._width, _] = self._img.shape
        self.FOV = FOV
        self.THETA = THETA
        self.PHI = PHI
    

    def GetEquirec(self,height,width):
        #
        # THETA is left/right angle, PHI is up/down angle, both in degree
        #

        equ_h = height
        equ_w = width
        equ_cx = (equ_w - 1) / 2.0
        equ_cy = (equ_h - 1) / 2.0

        wFOV = self.FOV
        hFOV = float(self._height) / self._width * wFOV

        w_len = np.tan(np.radians(wFOV / 2.0))
        h_len = np.tan(np.radians(hFOV / 2.0))


        x_map = np.ones([self._height, self._width], np.float32)
        y_map = np.tile(np.linspace(-w_len, w_len,self._width), [self._height,1])
        z_map = -np.tile(np.linspace(-h_len, h_len,self._height), [self._width,1]).T

        print(z_map[0])

        D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
        xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)
        print(xyz[0,:,2])
        
        y_axis = np.array([0.0, 1.0, 0.0], np.float32)
        z_axis = np.array([0.0, 0.0, 1.0], np.float32)
        [R1, _] = cv2.Rodrigues(z_axis * np.radians(self.THETA))
        [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-self.PHI))

        xyz = xyz.reshape([self._height * self._width, 3]).T
        xyz = np.dot(R1, xyz)
        xyz = np.dot(R2, xyz).T
        lat = np.arcsin(xyz[:, 2])
        lon = np.arctan2(xyz[:, 1] , xyz[:, 0])

        lon = lon / np.pi * 180
        lat = -lat / np.pi * 180

        print(lat.reshape([self._height , self._width])[0])
        print(lon.reshape([self._height , self._width])[0])
        
        lon = (lon / 180 * equ_cx + equ_cx).astype(np.int)
        lat = (lat / 90  * equ_cy + equ_cy).astype(np.int)
        coordinate = (lat,lon)

        x_map = np.repeat(np.arange(self._height), self._width)
        y_map = np.tile(np.arange(self._width), self._height)

        blank_map_x = np.zeros((height,width))
        blank_map_y = np.zeros((height,width))
        mask = np.zeros((height,width,3))

        blank_map_x[coordinate] = x_map
        blank_map_y[coordinate] = y_map
        mask[coordinate] = [1,1,1]

        # print(lat.reshape([self._height, self._width]))
        # print(lon.reshape([self._height, self._width])[-1,1910:1930])


        persp = cv2.remap(self._img, blank_map_y.astype(np.float32), blank_map_x.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        
        persp = persp * mask
        
        return persp , mask
        






