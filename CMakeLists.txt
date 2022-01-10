cmake_minimum_required(VERSION 3.0.2)
project(habitat_sim_ros)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  geometry_msgs
  message_generation
  nav_msgs
  rosgraph_msgs
  rospy
  sensor_msgs
  tf2
  tf2_ros
)

add_service_files(
  FILES
  Reset.srv
  LoadScene.srv
  RespawnAgent.srv
)

generate_messages(
  DEPENDENCIES
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS
  cv_bridge
  geometry_msgs
  message_runtime
  nav_msgs
  rosgraph_msgs
  rospy
  sensor_msgs
  tf2
  tf2_ros
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  scripts/habitat_sim_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY
  launch
  param
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
