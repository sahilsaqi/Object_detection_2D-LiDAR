  GNU nano 4.8                                                                       CMakeLists.txt                                                                                 cmake_minimum_required(VERSION 3.0.2)
project(object_detection_2)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  message_generation
)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs message_runtime
)

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/object_detection_with_slam.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install other files
install(DIRECTORY
  launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
