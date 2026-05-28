cmake_minimum_required(VERSION 3.8)
project(adlap_tool_control)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(yaml-cpp REQUIRED)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  ament_add_gtest(test_angle_conversion
    test/test_angle_conversion.cpp
    test/motor_controller_stub.cpp
    src/instrument_controller.cpp
    src/adlap_gearbox.cpp
    src/keyboard_reader.cpp
  )

  if(TARGET test_angle_conversion)
    target_include_directories(test_angle_conversion PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    ament_target_dependencies(test_angle_conversion rclcpp std_msgs geometry_msgs)
    target_link_libraries(test_angle_conversion yaml-cpp)
  endif()
endif()

add_executable(controller 
  src/node.cpp
  src/instrument_controller.cpp
  src/manual_adjustment.cpp
  src/adlap_gearbox.cpp
  src/serial_port.cpp
  src/keyboard_reader.cpp
  src/motor_controller.cpp
  src/motor_thread.cpp
)
  
ament_target_dependencies(controller rclcpp std_msgs geometry_msgs)
target_link_libraries(controller yaml-cpp)

# Include directories
target_include_directories(controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# Install headers
install(
  DIRECTORY include/
  DESTINATION include/${PROJECT_NAME}
)

install(TARGETS
  controller
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/tool_reader_node.py
  scripts/setup_test_runner_node.py
  scripts/plot_trial.py
  scripts/gearbox_digital_twin.py
  scripts/gearbox_state_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
