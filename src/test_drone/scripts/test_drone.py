import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry, SensorCombined, VehicleRatesSetpoint, TrajectorySetpoint, VehicleCommand, VehicleStatus, OffboardControlMode
from control_msgs.msg import DynamicJointState
from trajectory_msgs.msg import MultiDOFJointTrajectory
from std_msgs.msg import Float64MultiArray
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import torch
from vehicle import Vehicle
from model import ModelLoader, LoadONNX
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import csv
from collections import deque

def gravity_in_frd(px4_quat, vector):
    """
    Convert gravity vector into PX4 FRD body frame, given PX4 quaternion.   

    PX4 quaternion = rotation from BODY(FRD) → NED.

    Args:
        px4_quat: [w, x, y, z] quaternion (BODY → NED)
        g: gravity magnitude

    Returns:
        np.array([gx, gy, gz]) in FRD frame
    """

    # PX4 gives quaternion as [w, x, y, z]   (scalar-first)
    w, x, y, z = px4_quat

    # SciPy expects [x, y, z, w] (vector-first)
    scipy_quat = np.array([x, y, z, w])

    # Create BODY→NED rotation
    R_body_to_ned = R.from_quat(scipy_quat)

    # To express a NED vector in BODY frame, invert the rotation
    R_ned_to_body = R_body_to_ned.inv()

    # Gravity in NED (+Z downward)
    g_ned = vector

    # Transform gravity into FRD frame
    g_frd = R_ned_to_body.apply(g_ned)

    return g_frd


class Control(Node):
    
    def __init__(self, target_pos):
        super().__init__('drone_control')   # Node name
        self.get_logger().info("Control has started!")        
        self.vehicle = Vehicle(self)

        self.model = LoadONNX("models/13-21-06-58.onnx")

        self.target_pos = np.array([0.0, 0.0, 2.5])
        self.target_quat = np.array([0.0, 0.0, 0.0, 1.0])   #[x, y, z, w]

        self.sim_dt = 0.02
        self.i = 0

        self.mass = 2.433

        # ---- Examples: Create a timer ----
        timer_period = 0.02  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.trajectory_sub = self.create_subscription(
            MultiDOFJointTrajectory,
            '/command/trajectory',  # Replace with your actual topic name
            self.trajectory_callback,
            10
        )

        self.trajectory_data = []  # Store [timestamp, target_x, target_y, target_z, actual_x, actual_y, actual_z]
        self.position_errors = deque(maxlen=1000)  # Store recent errors for statistics
        self.csv_filename = f'trajectory_tracking_{int(self.get_clock().now().nanoseconds / 1e9)}.csv'
        
        # Initialize CSV file with headers
        with open(self.csv_filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'target_x', 'target_y', 'target_z', 
                           'actual_x', 'actual_y', 'actual_z', 
                           'error_x', 'error_y', 'error_z', 'error_magnitude'])
        
        self.get_logger().info(f'Saving trajectory data to: {self.csv_filename}')
        
        # Timer for statistics printing (every 5 seconds)
        self.stats_timer = self.create_timer(5.0, self.print_statistics)
        self.last_stats_print = self.get_clock().now()

        # self.maxthrst = 9.413
        # self.maxthrst = 7.25      #hover servo output = 840, hover pwm = 0.81176. for 2.4 kgs -> max_thr = 7.25144
        # self.maxthrst = 6.81
        self.maxthrst = 7.43    #hover servo output = 832.5, hover pwm = 080294. for 2.433 kgs -> max_thr = mg/4*hover_pwm =7.43


    def trajectory_callback(self, msg):
        """Callback to receive and store trajectory points."""
        if len(msg.points) > 0:
            # Get the first trajectory point
            point = msg.points[0]
            
            # Extract position (from transforms[0].translation)
            if len(point.transforms) > 0:
                transform = point.transforms[0]
                self.target_pos[0] = transform.translation.x/10
                self.target_pos[1] = transform.translation.y/10
                self.target_pos[2] = transform.translation.z/10
                
                self.get_logger().info(
                    f'Received target: x={self.target_x:.3f}, '
                    f'y={self.target_y:.3f}, z={self.target_z:.3f}'
                )

    def get_thrust(self, acc_z):

        force_z = (acc_z + 9.81) * self.mass
        norm_thrust = force_z/(self.maxthrst*4)

        return np.clip(norm_thrust, 0.0, 1.0)

    def get_control(self):

        drone_pos = np.array(self.vehicle.pos.copy())
        drone_linvel = np.array(self.vehicle.LV.copy())
        # r = R.from_quat(self.vehicle.quats)

        ned_to_enu = np.array([[0, 1, 0],    #this is a symmetric matrix (A)T = A
                               [1, 0, 0],
                               [0, 0, -1]])
        
        # r_lfu_enu = ned_to_enu @ r.as_matrix() @ ned_to_enu.T    #ned_to_enu = rfu_to_frd  #the correct frame is rfu not lfu

        joint_pos = np.array(self.vehicle.joint_pos)
        joint_vel = np.array(self.vehicle.joint_vel)

        drone_quat = np.array(self.vehicle.quats)
        drone_R_frd_ned = R.from_quat(drone_quat)
        drone_R_rfu_enu = ned_to_enu @ drone_R_frd_ned.as_matrix() @ ned_to_enu
        drone_R6 = self.R6_from_9(drone_R_rfu_enu)
        drone_quat = R.from_matrix(drone_R_rfu_enu).as_quat()

        ee_pos, r_ee = self.calculate_ee_state(drone_pos, drone_quat, joint_pos)
        ee_matrix = r_ee.as_matrix()

        drone_rel_pos = self.target_pos - drone_pos
        ee_rel_pos = self.target_pos - ee_pos

        target_R = R.from_quat(self.target_quat)

        target_R_rel_ee = ee_matrix.T@ target_R.as_matrix()
        target_R_flat = target_R_rel_ee.flatten()

        obs = np.concatenate([drone_rel_pos/10, drone_R6, drone_linvel/10, self.vehicle.AV, ee_rel_pos/10, target_R_flat, joint_pos, joint_vel]).astype(np.float32)

        obs_input = np.expand_dims(obs, axis=0)
        # obs = np.array([[5.6962529e-03, -6.5228711e-03,  1.0015297e+00,  9.9984783e-01, 1.7444085e-02, -1.6304986e-06, -1.7444085e-02,  9.9984783e-01, 2.7210058e-06,  1.6777152e-06, -2.6921496e-06,  1.0000000e+00, 4.7197806e-05, -8.3191735e-05, -9.5338997e-05, -3.4885236e-06, -1.6702197e-06,  3.5352804e-04,  1.8199615e+00]]).astype(np.float32)
        print(f"OBSERVATION: {obs}")
        raw_action = self.model.predict(obs_input)
        # print(f"RAW_ACTION: {raw_action}")

        if hasattr(raw_action, 'numpy'):
            raw_action = raw_action.numpy()
        
        if raw_action.ndim > 1:
            raw_action = raw_action.squeeze()
        #print(raw_action.shape)
        # print(raw_action)

        action = np.zeros_like(raw_action)
        actions_clipped = np.clip(raw_action, -1, 1)

        action[0] = actions_clipped[0]*5.0
        action[1] = actions_clipped[1]*5.0
        action[2] = actions_clipped[2]*9.81
        action[3] = actions_clipped[3]*8.0
        action[4] = actions_clipped[4]*8.0
        action[5] = actions_clipped[5]*6.0
        action[6] = actions_clipped[6]*np.pi/2
        action[7] = actions_clipped[7]*1.57 -1.57
        action[8] = actions_clipped[8]*2.094
        action[9] = actions_clipped[9]*3.14

       
        # norm_thrust = self.get_thrust(action[0])

        # ctbr = np.array([norm_thrust, action[1], action[2], action[3]])
        # joints = np.array([action[4], action[5], action[6]]) 

        ctbr = np.array([action[0], action[1], action[2], action[3] ,action[4], action[5], action[6]])
        joints = np.array([action[7], action[8], action[9]]) 

        return ctbr, joints
    
    def calculate_ee_state(self, drone_pos, drone_quat, joint_angles):
        """
        Calculates End Effector position based on the X500 PX4 4-Link Arm URDF.
        
        Frames:
        - drone_pos: ENU (East-North-Up)
        - drone_quat: Transforms Local RFU -> World ENU
        - local_offset: Calculated natively in RFU (Right-Forward-Up)
        
        URDF Pitch Offsets:
        - Base Mount is pitched 180 deg.
        - Joint 1 is pitched 90 deg (1.57 rad).
        - Total static pitch = 270 deg (1.5 * PI). 
        - At theta=0, arm points horizontally backward.
        """
        theta1 = joint_angles[0]
        theta2 = joint_angles[1]
        # Safely handle theta3 if your array only has 2 joints right now
        theta3 = joint_angles[2] if len(joint_angles) > 2 else 0.0

        # URDF Lengths and Offsets
        j1_x = 0.03       # Small offset in the elbow
        j1_z = 0.23682    # Length of upper arm
        j2_z = 0.21599    # Length of lower arm
        
        # Base mount is Z = -0.025.
        # L1 offset is Z = 0.03796
        # Total Base Z Offset = -0.025 - 0.03796 = -0.06296
        base_z_offset = -0.06296
        
        # --- POSITION (Local RFU Frame) ---
        # Arm swings in the mid-sagittal plane, so Right (X) is 0
        ee_x_local = 0.0
        
        # Forward/Backward extension (Y in RFU)
        ee_y_local = j1_x * math.sin(theta1) - j1_z * math.cos(theta1) - j2_z * math.cos(theta1 + theta2)
        
        # Up/Down extension (Z in RFU)
        ee_z_local = base_z_offset + j1_x * math.cos(theta1) + j1_z * math.sin(theta1) + j2_z * math.sin(theta1 + theta2)
        
        # Construct Local Offset Vector (Right, Forward, Up)
        local_offset = np.array([ee_x_local, ee_y_local, ee_z_local])
        
        # Apply Drone Orientation to get World (ENU) Position
        r_drone = R.from_quat(drone_quat) 
        world_ee_pos = drone_pos + r_drone.apply(local_offset)

        # --- ORIENTATION ---
        # A. Swing Rotation
        # URDF rotates around the Y axis (Left). In RFU, Left is the Negative X-axis.
        # Total static offset = 1.5 * PI (270 degrees).
        swing_angle_x = -(1.5 * math.pi + theta1 + theta2)
        r_swing = R.from_rotvec([swing_angle_x, 0.0, 0.0])
        
        # B. Twist Rotation (Rotation around local +Z axis by theta3)
        r_twist = R.from_rotvec([0.0, 0.0, theta3])

        # Combine rotations: Swing first, then Twist
        r_arm_local = r_swing * r_twist

        # Combine with Drone Rotation: (Drone Rotation) * (Arm Rotation)
        r_ee_world = r_drone * r_arm_local
        
        # Returns the world position, and the 3x3 Rotation Matrix
        return world_ee_pos, r_ee_world

    def R6_from_9(self, R_matrix):
        """
        Converts 3x3 Rotation Matrix to 6D representation.
        Takes first two columns of the rotation matrix.
        Output shape: (6,)
        """
        # R_matrix is 3x3. 
        # col1 = R_matrix[:, 0], col2 = R_matrix[:, 1]
        r6 = np.concatenate([R_matrix[:, 0], R_matrix[:, 1]])
        return r6

    def get_flightmare_policy(self, raw_action):

        # print(self.vehicle.pos)
        # print(self.target)
        # goal_local = self.target - self.vehicle.pos

        # r = R.from_quat(self.vehicle.quats)
        # euler = r.as_matrix()

        # euler = np.reshape(euler, (9,))

        # obs = np.concatenate([goal_local, euler, self.vehicle.LV, self.vehicle.AV])
        # # obs[3] =  0.0
        # obs = obs.astype(np.float32)
        # obs = np.expand_dims(obs, axis=0)
        # print(f"observation: {obs}")
        # # obs = np.array([[ 0.20710441,  0.17971317,  0.17334747,  0.31125432,  0.01186327,  0.01035172,
        #     # -0.10929969, -0.04562206,  0.02839059, -0.04129951,  0.01239144, -0.022663  ]]).astype(np.float32)
        # raw_action = self.model.predict(obs)
            
        print(f"RAW ACTION {raw_action}")

        action_processed = raw_action*self.std + self.mean

        print(f"PROCCESSED : {action_processed}")

        print("STD: ", self.std)

        normalized_thrust = action_processed / self.total_thrust

        print(f"normalized thrust : {normalized_thrust}")

        return normalized_thrust 

    def timer_callback(self):

        if not self.vehicle.received_odom:
            self.get_logger().info("Waiting for first odometry message from PX4...", throttle_duration_sec=2)
            return

        # if not self.vehicle.offboard:
        #     self.vehicle.enable_offboard()
        #     return
        
        ctbr, joint_setpoints = self.get_control()
        # print(f"ctbr: {ctbr}")
        # print(f"joint setpoints: {joint_setpoints}")
        # print("\n")

        # self.vehicle.offboard_control()

        self.vehicle.publish_ctbr_asmc(ctbr)
        self.vehicle.publish_joints(joint_setpoints)

        curr_pos = self.vehicle.pos.copy()
        error_x = self.target_pos[0] - curr_pos[0]  # Assuming obs[0] is current x
        error_y = self.target_pos[1] - curr_pos[1]  # Assuming obs[1] is current y
        error_z = self.target_pos[2] - curr_pos[2]  # Assuming obs[2] is current z
        error_magnitude = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        
        # Store error for statistics
        self.position_errors.append(error_magnitude)
        
        # Get current timestamp
        current_time = self.get_clock().now().nanoseconds / 1e9
        
        # Save to CSV
        with open(self.csv_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_time,
                - self.target_pos[0], self.target_pos[1], self.target_pos[2],
                curr_pos[0], curr_pos[1], curr_pos[2],
                error_x, error_y, error_z,
                error_magnitude
            ])

    def print_statistics(self):
        """Print tracking statistics every few seconds"""
        if len(self.position_errors) > 0:
            errors_array = np.array(self.position_errors)
            rmse = np.sqrt(np.mean(errors_array**2))
            mae = np.mean(errors_array)
            max_error = np.max(errors_array)
            min_error = np.min(errors_array)
            std_error = np.std(errors_array)
            
            self.get_logger().info(
                f'\n=== Tracking Statistics (last {len(self.position_errors)} samples) ===\n'
                f'RMSE: {rmse:.4f} m\n'
                f'MAE:  {mae:.4f} m\n'
                f'Max Error: {max_error:.4f} m\n'
                f'Min Error: {min_error:.4f} m\n'
                f'Std Dev: {std_error:.4f} m\n'
                f'========================================='
            )

    def destroy_node(self):
        """Called when node is shut down - print final statistics"""
        self.get_logger().info(f'\nTrajectory data saved to: {self.csv_filename}')
        if len(self.position_errors) > 0:
            self.print_statistics()
        super().destroy_node()
        

def main(args=None):
    rclpy.init(args=args)
    target = [0.5,0,0]
    node = Control(target)
    try:
        rclpy.spin(node)     # Keep node alive
    except KeyboardInterrupt:
        pass  # Handle Ctrl+C gracefully
    finally:
        # This will always execute, even on Ctrl+C
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
