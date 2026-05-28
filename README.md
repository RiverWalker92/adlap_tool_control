# AdLap Tool Control


ROS2 Node for connecting to the AdLap gearbox over serial and driving the instrument.
Subscribes to message of the form of array in radians: 
- Euler angles: [roll, pitch, yaw, gipper_angle]
- Joint angles: [shaft_roll, bend, tip_rotation, articulation]

The right instrument is used for testing. To start the node run:


```bash
ros2 launch adlap_tool_control right.launch.py
```


```
ros2 launch adlap_tool_control circular_trajectory_node.launch.py
```

**Testing**

The package includes two offline angle-conversion exporters (using the stubbed motor controller) that generate CSV files for inspection in a notebook with pandas and matplotlib:

- `angle_conversion_export`: Sweeps each Euler angle across its full range
- `joint_angle_conversion_export`: Drives each joint angle with a sinusoidal sweep (0 to π/4)

- Build the package (with tests enabled) and run the exporters directly from the workspace build directory:

```bash
# Build the package and test-targets
cd /home/roel/ws_moveit
colcon build --packages-select adlap_tool_control --cmake-args -DBUILD_TESTING=ON

# Create data directory for CSV output
mkdir -p src/adlap_tool_control/data

# Euler-angle exporter (writes CSV using default samples configured in-source)
./build/adlap_tool_control/angle_conversion_export --output src/adlap_tool_control/data/angle_conversion_export.csv

# Joint-angle sinusoidal exporter (one-cycle sinusoid 0..pi/4)
./build/adlap_tool_control/joint_angle_conversion_export --output src/adlap_tool_control/data/joint_angle_conversion_export.csv

# Single line:
cd /home/roel/ws_moveit && colcon build --packages-select adlap_tool_control --cmake-args -DBUILD_TESTING=ON && ./build/adlap_tool_control/angle_conversion_export --output src/adlap_tool_control/data/angle_conversion_export.csv && ./build/adlap_tool_control/joint_angle_conversion_export --output src/adlap_tool_control/data/joint_angle_conversion_export.csv
```

- Or run via `ros2 run` (make sure to source your workspace install before running and pass program args after `--`):

```bash
# Build the package and test-targets (if not already done)
cd /home/roel/ws_moveit
colcon build --packages-select adlap_tool_control --cmake-args -DBUILD_TESTING=ON

# Source the install and run
source /home/roel/ws_moveit/install/setup.bash
mkdir -p src/adlap_tool_control/data
ros2 run adlap_tool_control angle_conversion_export -- --output src/adlap_tool_control/data/angle_conversion_export.csv
ros2 run adlap_tool_control joint_angle_conversion_export -- --output src/adlap_tool_control/data/joint_angle_conversion_export.csv
```

Notes:
- The exporters are test-only utilities and use the stubbed motor controller to avoid hardware access.
- Output CSV paths shown above are workspace-relative so they appear in the VS Code explorer. Feel free to change them to `/tmp` or another folder.

