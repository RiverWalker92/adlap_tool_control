# AdLap Tool Control


ROS2 Node for connecting to the AdLap gearbox over serial and driving the instrument.
Subscribes to message of the form of array in radians: 
- Euler angles: [roll, pitch, yaw, gipper_angle]
- Joint angles: [shaft_roll, bend, tip_rotation, articulation]


**Test Usage**
- **Build package (single line, run from workspace root):**

	```bash
	colcon build --packages-select adlap_tool_control --cmake-args -DBUILD_TESTING=ON
	```

- **Build package and run gtest (single line, run from workspace root):**
    ```bash
	colcon build --packages-select adlap_tool_control --cmake-args -DBUILD_TESTING=ON && colcon test --packages-select adlap_tool_control --event-handlers console_direct+ --ctest-args -R test_angle_conversion --output-on-failure
	```

- **Run the angle-conversion gtest only (single line, run from workspace root):**

	```bash
	colcon test --packages-select adlap_tool_control --event-handlers console_direct+ --ctest-args -R test_angle_conversion --output-on-failure
	```

Notes:
- Run these commands from your workspace root (the folder that contains `src/`).
- The build command enables package tests so the gtest target is available.
- The test command runs only the `test_angle_conversion` test and prints failures to the console.


