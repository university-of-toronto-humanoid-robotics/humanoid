cmake_minimum_required(VERSION 3.8)
project(bringup)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY config launch rviz
  DESTINATION share/${PROJECT_NAME}
)

install(
  DIRECTORY ../../../description/g1
  DESTINATION share/${PROJECT_NAME}/description
)

ament_package()
