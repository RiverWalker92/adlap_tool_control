# AdLap Tool Control

ROS2 Node for connecting to the AdLap gearbox over serial and driving the instrument.
Subscribes to message of the form of array in radians: [roll, pitch, yaw, gipper_angle] 


## Test utils
There are a couple of nodes to test trajectories.

`tool_controller_node` allows to you to send different trajectories to each of the DoFs (roll, pitch, yaw, aperture). There are three modes of trajectories: constant value, triangular motion and sinusoidal motion. Each of the modes accept different parameters. Check `tool_params.yaml` in `config` folder to see how to configure.

```
ros2 launch adlap_tool_control tool_controller_node.launch.py
```

`circular_trajectory_node` allows to you to send a circular trajecotry defined in pitch/yaw. Amplitude and frequency of the movement can be changed in `circular_params.yaml` in `config` folder. 

```
ros2 launch adlap_tool_control circular_trajectory_node.launch.py
```
