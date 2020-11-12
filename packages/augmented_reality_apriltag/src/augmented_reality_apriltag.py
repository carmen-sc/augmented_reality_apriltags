#!/usr/bin/env python3
import numpy as np
import os
import math
import cv2
import copy
import rospy
import rospkg 
import yaml
import sys

from renderClass import Renderer
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, CameraInfo
from sensor_msgs.srv import SetCameraInfo, SetCameraInfoResponse
from cv_bridge import CvBridge, CvBridgeError

from dt_apriltags import Detector
from augmented_reality_helper_module import Helpers

"""

This is a template that can be used as a starting point for the CRA1 exercise.
You need to project the model file in the 'models' directory on an AprilTag.
To help you with that, we have provided you with the Renderer class that render the obj file.

"""

class ARNode(DTROS):

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(ARNode, self).__init__(node_name=node_name,node_type=NodeType.GENERIC)
        self.veh = rospy.get_namespace().strip("/")

        rospack = rospkg.RosPack()
        # Initialize an instance of Renderer giving the model in input.
        self.renderer = Renderer(rospack.get_path('augmented_reality_apriltag') + '/src/models/duckie.obj')

        # Load calibration files
        self._res_w = 640
        self._res_h = 480
        self.calibration_call()

        # Initialize classes
        self.at_detec = Detector()
        self.helper = Helpers(self.current_camera_info)
        self.bridge = CvBridge()

        # Subscribe to the compressed camera image
        self.cam_image = rospy.Subscriber(f'/{self.veh}/camera_node/image/compressed', CompressedImage, self.callback, queue_size=10)
        
        # Publishers
        self.aug_image = rospy.Publisher(f'/{self.veh}/{node_name}/image/compressed', CompressedImage, queue_size=10)
        

    def callback(self, image):
        """ 
            Callback method that ties everything together
        """
        # convert the image to cv2 format
        image = self.bridge.compressed_imgmsg_to_cv2(image)

        #undistort the image
        image = self.helper.process_image(image)
        
        # detect AprilTags
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tags = self.at_detec.detect(gray, estimate_tag_pose=False, camera_params=None, tag_size=None)

        for tag in tags:
            # estimate homography
            homography = tag.homography

            # calculate projection matrix
            proj_mat = self.projection_matrix(self.current_camera_info.K, homography)

            # rendering magics
            image = self.renderer.render(image, proj_mat)

        #publish image to new topic
        self.aug_image.publish(self.bridge.cv2_to_compressed_imgmsg(image))



    
    def projection_matrix(self, intrinsic, homography):
        """
            Write here the compuatation for the projection matrix, namely the matrix
            that maps the camera reference frame to the AprilTag reference frame.
        """
        # format the K matrix
        K = np.reshape(np.array(intrinsic), (3,3))

        # calculate R_t
        R_t = np.matmul(np.linalg.inv(K), homography)
        
        # normalization parameters R_t
        n_r1 = np.linalg.norm(R_t[:, 0])
        n_r2 = np.linalg.norm(R_t[:, 1])

        # calculate R_3 and rebuild R_t
        R = np.array([R_t[:, 0], R_t[:, 1], (n_r1+n_r2)/2*np.cross(R_t[:, 0]/n_r1, R_t[:, 1]/n_r2)])
        R_t = np.vstack((R, R_t[:, 2]))

        # calculate the projection matrix
        proj_mat = np.matmul(K, np.transpose(R_t))

        return proj_mat

    
    def readYamlFile(self,fname):
        """
            Reads the 'fname' yaml file and returns a dictionary with its input.

            You will find the calibration files you need in:
            `/data/config/calibrations/`
        """
        with open(fname, 'r') as in_file:
            try:
                yaml_dict = yaml.load(in_file)
                return yaml_dict
            except yaml.YAMLError as exc:
                self.log("YAML syntax error. File: %s fname. Exc: %s"
                         %(fname, exc), type='fatal')
                rospy.signal_shutdown()
                return


    def calibration_call(self):
        # copied from the camera driver (dt-duckiebot-interface)

        # For intrinsic calibration
        self.cali_file_folder = '/data/config/calibrations/camera_intrinsic/'
        self.frame_id = rospy.get_namespace().strip('/') + '/camera_optical_frame'
        self.cali_file = self.cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"

        # Locate calibration yaml file or use the default otherwise
        if not os.path.isfile(self.cali_file):
            self.logwarn("Calibration not found: %s.\n Using default instead." % self.cali_file)
            self.cali_file = (self.cali_file_folder + "default.yaml")

        # Shutdown if no calibration file not found
        if not os.path.isfile(self.cali_file):
            rospy.signal_shutdown("Found no calibration file ... aborting")

        # Load the calibration file
        self.original_camera_info = self.load_camera_info(self.cali_file)
        self.original_camera_info.header.frame_id = self.frame_id
        self.current_camera_info = copy.deepcopy(self.original_camera_info)
        self.update_camera_params()
        self.log("Using calibration file: %s" % self.cali_file)


        # extrinsic calibration
        self.ex_cali_file_folder = '/data/config/calibrations/camera_extrinsic/'
        self.ex_cali_file = self.ex_cali_file_folder + rospy.get_namespace().strip("/") + ".yaml"
        self.ex_cali = self.readYamlFile(self.ex_cali_file)
        
        # define homography
        self.homography = self.ex_cali["homography"]

    def update_camera_params(self):
        """ Update the camera parameters based on the current resolution.
        The camera matrix, rectification matrix, and projection matrix depend on
        the resolution of the image.
        As the calibration has been done at a specific resolution, these matrices need
        to be adjusted if a different resolution is being used.
        TODO: Test that this really works.
        """

        scale_width = float(self._res_w) / self.original_camera_info.width
        scale_height = float(self._res_h) / self.original_camera_info.height

        scale_matrix = np.ones(9)
        scale_matrix[0] *= scale_width
        scale_matrix[2] *= scale_width
        scale_matrix[4] *= scale_height
        scale_matrix[5] *= scale_height

        # Adjust the camera matrix resolution
        self.current_camera_info.height = self._res_h
        self.current_camera_info.width = self._res_w

        # Adjust the K matrix
        self.current_camera_info.K = np.array(self.original_camera_info.K) * scale_matrix

        # Adjust the P matrix (done by Rohit)
        scale_matrix = np.ones(12)
        scale_matrix[0] *= scale_width
        scale_matrix[2] *= scale_width
        scale_matrix[5] *= scale_height
        scale_matrix[6] *= scale_height
        self.current_camera_info.P = np.array(self.original_camera_info.P) * scale_matrix

    @staticmethod
    def load_camera_info(filename):
        """Loads the camera calibration files.
        Loads the intrinsic camera calibration.
        Args:
            filename (:obj:`str`): filename of calibration files.
        Returns:
            :obj:`CameraInfo`: a CameraInfo message object
        """
        with open(filename, 'r') as stream:
            calib_data = yaml.load(stream)
        cam_info = CameraInfo()
        cam_info.width = calib_data['image_width']
        cam_info.height = calib_data['image_height']
        cam_info.K = calib_data['camera_matrix']['data']
        cam_info.D = calib_data['distortion_coefficients']['data']
        cam_info.R = calib_data['rectification_matrix']['data']
        cam_info.P = calib_data['projection_matrix']['data']
        cam_info.distortion_model = calib_data['distortion_model']
        return cam_info
        


    def onShutdown(self):
        super(ARNode, self).onShutdown()


if __name__ == '__main__':
    # Initialize the node
    camera_node = ARNode(node_name='augmented_reality_apriltag_node')
    # Keep it spinning to keep the node alive
    rospy.spin()