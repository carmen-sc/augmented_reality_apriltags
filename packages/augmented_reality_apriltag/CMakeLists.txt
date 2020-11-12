cmake_minimum_required(VERSION 2.8.3)
project(augmented_reality_apriltag)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  cv_bridge
  duckietown_msgs
  image_geometry
)

catkin_package()