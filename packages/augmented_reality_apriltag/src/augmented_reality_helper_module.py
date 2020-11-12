#!/usr/bin/env python3
import numpy as np
import os
import cv2
from image_geometry import PinholeCameraModel


class Helpers:

    def __init__(self, camera_info):
        self.ci = camera_info
        self.pcm = PinholeCameraModel()
        self.pcm.fromCameraInfo(self.ci)
        self._rectify_inited = False
        self._distort_inited = False
               
    def process_image(self, cv_image_raw, interpolation=cv2.INTER_NEAREST):
        ''' Undistort an image.
            To be more precise, pass interpolation= cv2.INTER_CUBIC
        '''
        if not self._rectify_inited:
            self._init_rectify_maps()

        cv_image_rectified = np.empty_like(cv_image_raw)
        res = cv2.remap(cv_image_raw, self.mapx, self.mapy, interpolation,
                        cv_image_rectified)
        return res

    def _init_rectify_maps(self):
        W = self.pcm.width
        H = self.pcm.height
        mapx = np.ndarray(shape=(H, W, 1), dtype='float32')
        mapy = np.ndarray(shape=(H, W, 1), dtype='float32')
        mapx, mapy = cv2.initUndistortRectifyMap(self.pcm.K, self.pcm.D, self.pcm.R,
                                                 self.pcm.P, (W, H),
                                                 cv2.CV_32FC1, mapx, mapy)
        self.mapx = mapx
        self.mapy = mapy
        self._rectify_inited = True

    
