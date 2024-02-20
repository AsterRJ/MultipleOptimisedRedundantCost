cmake_minimum_required(VERSION 3.0.2)
project(kmriiwa_nullspace)

## Compile as C++11, supported in ROS Kinetic and newer
#add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  roscpp
  rospy
  sensor_msgs
  std_msgs
  trajectory_msgs
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES kmriiwa_nullspace
#  CATKIN_DEPENDS geometry_msgs roscpp rospy sensor_msgs std_msgs trajectory_msgs
#  DEPENDS system_lib
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)



include_directories (/usr/local/include/eigen3)

#add_library(${PROJECT_NAME} src/calculus_lib/nullspace_calcs.cpp)
#target_link_libraries(${PROJECT_NAME}  ${catkin_LIBRARIES})



add_executable(${PROJECT_NAME}_node src/nullspace_server.cpp)
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES})
#add_dependencies()







