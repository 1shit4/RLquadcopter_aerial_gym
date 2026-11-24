# Aerial-Manipulator RL Policy Inference with PX4–Gazebo

**ROS 2 interface for running goal-conditioned aerial-manipulator policies trained in Aerial Gym with a PX4–Gazebo vehicle and robotic arm.**

This repository supports the [Robust Goal-Directed Aerial Manipulation project](https://github.com/1shit4/aerial-manipulation-rl-control). The corresponding [Aerial Gym repository](https://github.com/1shit4/aerial_gym_simulator) implements goal-directed control of an aerial manipulator using reinforcement learning and a disturbance-compensating inner-loop controller. This workspace implements the **inference-side interface** between the learned policy, PX4 vehicle feedback, the external flight controller, and the arm's joint controller.

## Policy Inference Interface

The main control script, [`test_drone.py`](src/test_drone/scripts/test_drone.py), constructs the policy observation from PX4 vehicle feedback, simulated arm-joint states, and a desired end-effector target. An exported ONNX policy generates control actions, which are rescaled and sent to the vehicle and arm control interfaces. The main loop is configured with a **20 ms update period (50 Hz)**.

### Observation space

The interface constructs a **33-dimensional observation**, matching the aerial-manipulator policy configuration:

| Observation component | Dimensions |
| --- | ---: |
| Vehicle position relative to the goal | 3 |
| Vehicle attitude (6D rotation representation) | 6 |
| Vehicle linear and angular velocities | 6 |
| End-effector position relative to the goal | 3 |
| End-effector orientation error (flattened rotation matrix) | 9 |
| Arm-joint positions and velocities | 6 |
| **Total** | **33** |

End-effector pose is calculated from the vehicle pose and manipulator-joint configuration. The implementation also converts vehicle position, attitude, and angular-rate information between the **PX4 NED/FRD** and **training ENU/RFU** coordinate conventions.

### Policy action space

The interface processes **10 normalized policy outputs**:

| Policy command | Dimensions | Control interface |
| --- | ---: | --- |
| Desired vehicle acceleration | 3 | External flight controller |
| Desired body rates | 3 | External flight controller |
| Desired yaw | 1 | External flight controller |
| Desired arm-joint positions | 3 | Simulated arm-position controller |
| **Total** | **10** | |

The seven vehicle commands are sent together to the external flight-control interface; the three joint-position references are sent to the arm controller. This action space is **acceleration + body rates + yaw + joint positions**, rather than the four-element collective-thrust/body-rate (CTBR) action used in the earlier standalone-quadrotor experiments.

The corresponding disturbance-compensating flight-control laws are documented in the [quadrotor inner-loop control repository](https://github.com/1shit4/quadcopter_inner_loop_control).

## Trajectory and Tracking Utilities

[`traj.py`](src/test_drone/scripts/traj.py) provides reference-generation utilities for hover, straight-line, circular, square, figure-eight, vertical, and helical motion. The inference script also contains tools for recording target and measured vehicle positions, computing position errors, and logging trajectories to CSV for subsequent analysis.

## Code Organization

| File | Purpose |
| --- | --- |
| [`test_drone.py`](src/test_drone/scripts/test_drone.py) | Policy observation construction, end-effector forward kinematics, action processing, inference loop, and tracking logs |
| [`vehicle.py`](src/test_drone/scripts/vehicle.py) | PX4 state feedback, coordinate handling, vehicle-reference interface, and arm-joint commands |
| [`model.py`](src/test_drone/scripts/model.py) | ONNX Runtime policy loading, with an additional TorchScript model loader |
| [`traj.py`](src/test_drone/scripts/traj.py) | Reference-trajectory generation |
| [`transform.py`](src/test_drone/scripts/transform.py) | Coordinate-transformation utilities and experiments |

The workspace uses **ROS 2, PX4 ROS messages, Gazebo, NumPy, SciPy, PyTorch, and ONNX Runtime**. Running the inference interface requires a compatible exported policy, the corresponding PX4–Gazebo aerial-manipulator setup, and the external flight/arm controllers. The model path in the inference script is configurable for the local exported policy.

## Related Repositories

- [Aerial Manipulation — Project Overview](https://github.com/1shit4/aerial-manipulation-rl-control): research objective and index of experimental implementations.
- [Aerial Gym Simulator](https://github.com/1shit4/aerial_gym_simulator): goal-conditioned aerial-manipulator RL task, policy training, and controller-in-the-loop implementation.
- [Quadrotor Inner-Loop Control](https://github.com/1shit4/quadcopter_inner_loop_control): disturbance-compensating control algorithms and PX4 interfaces.
- [Flightmare Policy Inference](https://github.com/1shit4/RLQuadcopterControlFlightmare): related sim-to-sim policy-inference implementation for standalone quadrotor control.

*Research conducted collaboratively at the Robotics Research Center, IIIT Hyderabad.*
