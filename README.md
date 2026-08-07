Underwater Vehicle Simulation Environment for VIO

## Overview
This repository provides a simulation environment for underwater vehicle research and development, with a focus on Visual-Inertial Odometry (VIO) and sea modeling. The system is built on the Robot Operating System (ROS) and Gazebo simulation platform, integrating multiple underwater vehicle models with custom control algorithms and environmental representations.

## System Requirements
- **Operating System**: Ubuntu 20.04 LTS
- **ROS Distribution**: ROS Noetic Ninjemys
- **Simulation Platform**: Gazebo 11
- **Build System**: Catkin Tools


## Installation

### Prerequisites Installation
```bash
# Install ROS Noetic
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-noetic-desktop-full

# Install Gazebo 11
sudo apt install ros-noetic-gazebo-ros-pkgs ros-noetic-gazebo-ros2-control

# Install Catkin Tools
sudo apt install python3-catkin-tools python3-rosdep python3-rosinstall


# Clone the repository
git clone <repository-url>
cd AKIT-VIO

# Install workspace dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build the workspace
catkin build

# Source the workspace
echo "source $(pwd)/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

```
Note: This repository was initially configured 7 months ago. Some components may require updates for compatibility with current ROS/Gazebo versions. Refer to commands.txt for specific commands used during initial development.

Core Components
Vehicle Models
The repository includes multiple underwater vehicle description packages:
lauv_description: Contains URDF models with visual and collision geometry, inertial properties, and joint definitions
my_auv_description: Additional vehicle models with custom configurations
lauv_gazebo: Gazebo plugins, world files, and simulation parameters

Control Systems
lauv_control: Implements control algorithms including:
Thrust allocation
State estimation
Trajectory tracking
Stabilization controllers

Simulation Environment
uuv_simulator: Core simulation framework providing:
Hydrodynamic models
Underwater physics
Sensor simulation (IMU, pressure, DVL)
Environmental effects (currents, waves)

Coordinate Frames
The frames.gv and frames.pdf files illustrate the coordinate frame hierarchy used in the simulation:
world: Global reference frame
base_link: Vehicle body frame
imu_link: IMU sensor frame
camera_link: Camera frame for VIO

Development
Adding New Vehicle Models
Create new description package
Define URDF/XACRO model
Configure Gazebo plugins
Create launch files
Update CMakeLists.txt

Modifying Controllers
Locate controller source in lauv_control
Modify gains in configuration files
Rebuild package
Test with simulation

Extending VIO Capabilities
Add new sensor plugins
Implement data processing nodes
Configure transformer package
Update launch files

Additional Resources
ROS Noetic Documentation: http://wiki.ros.org/noetic
Gazebo Tutorials: http://gazebosim.org/tutorials
UUV Simulator: https://github.com/uuvsimulator/uuv_simulator
ROS Control: http://wiki.ros.org/ros_control
TF2 Library: http://wiki.ros.org/tf2

License
Apache-2.0

Authors
vinaya-53 - Initial work and repository setup

Acknowledgments
UUV Simulator development team
ROS and Gazebo communities
Open-source robotics contributors

