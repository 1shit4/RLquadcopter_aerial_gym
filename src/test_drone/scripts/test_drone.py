import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleOdometry, SensorCombined, VehicleRatesSetpoint, TrajectorySetpoint, VehicleCommand, VehicleStatus, OffboardControlMode
from trajectory_msgs.msg import MultiDOFJointTrajectory
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy
import torch
from vehicle import Vehicle
from model import ModelLoader, LoadONNX
import numpy as np
import math
from scipy.spatial.transform import Rotation as R
import csv
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation as R

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
        self.target = target_pos
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 0.15 #0.25
        self.i = 0
        self.yaw_rate = 0
        self.sim_dt = 0.02
        # self.model = ModelLoader("policy.pt")
        # self.model = LoadONNX("models/dr_none.onnx")
        # self.model = LoadONNX("models/dr_m0.5_a0.2.onnx")
        # self.model = LoadONNX("models/dr_m0.5_a_0.2_i_0.2.onnx")
        # self.model = LoadONNX("models/dr_m0.7_a_0.2_i_0.2.onnx")
        # self.model = LoadONNX("models/dr_m0.3_a0.1_i0.1.onnx")
        # self.model = LoadONNX("models/EpisodeV.onnx")
        self.model = LoadONNX("models/2upe.onnx")

        self.Gz = -9.81
        self.mass = 2.0
        self.inertia = np.array([[0.02166666666666667, 0, 0], 
                                [0, 0.02166666666666667, 0],
                                [0, 0, 0.04000000000000001]])
        
        self.kin_inv = np.array([[16.6, 0, 0], 
                                [0, 16.6, 0],
                                [0, 0, 5.0]])
        
        arm_l = 0.17
        kappa = 0.016
        
        t_BM = (arm_l * np.sqrt(0.5) *
                    np.array([
                        [ 1, -1, -1,  1],
                        [-1, -1,  1,  1],
                        [ 0,  0,  0,  0]
                    ])
                )
        
        self.allocation_matrix = np.vstack([ np.ones(4),
                                        t_BM[:2, :],
                                        kappa * np.array([1, -1, 1, -1])
                                    ])
        
        self.inverse_allocat = np.linalg.inv(self.allocation_matrix)
    
        # print(f'INERTIA: {self.inertia}')

        self.mean = np.ones(4)*(-self.Gz)
        self.std = np.ones(4)*(-self.Gz)*0.35

        self.total_thrust = self.mean*1.35

        self.i = 0

        #self.action_array = np.load("/home/vasudevan/Desktop/flightmare/flightrl/examples/actions_anh.npy")

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


        self.a = []
        # self.maxthrst = 9.413
        # self.maxthrst = 7.25      #hover servo output = 840, hover pwm = 0.81176. for 2.4 kgs -> max_thr = 7.25144
        self.maxthrst = 6.81

    def trajectory_callback(self, msg):
        """Callback to receive and store trajectory points."""
        if len(msg.points) > 0:
            # Get the first trajectory point
            point = msg.points[0]
            
            # Extract position (from transforms[0].translation)
            if len(point.transforms) > 0:
                transform = point.transforms[0]
                self.target_x = transform.translation.x/10
                self.target_y = transform.translation.y/10
                self.target_z = transform.translation.z/10
                
                self.get_logger().info(
                    f'Received target: x={self.target_x:.3f}, '
                    f'y={self.target_y:.3f}, z={self.target_z:.3f}'
                )

    def get_thrust(self, action):

        action[0] = action[0]*self.mass

        # err = action[1:4] - self.vehicle.AV
        # print(f"error {err}")
        # print(f"av {self.vehicle.AV}")

        # action[1:4] = self.inertia @ self.kin_inv @ err + np.cross(self.vehicle.AV, self.inertia @ self.vehicle.AV)

        # print(f"ACTION 1: {action}")
        # action[1:4] = [0, 0, 0]

        # action[3] = 0.0

        thrust = action
        # print(thrust.shape)

        # thrust = self.inverse_allocat @ action

        # print(f"ACTION 2: {thrust}")

        # thrust /= (9.81*2*self.mass/4)
        # thrust /= self.maxthrst

        thrust[0] = thrust[0]/(self.maxthrst*4)

        thrust[0] = np.clip(thrust[0], 0, 1.0)
        # print(f"ACTION 3: {thrust}")
        # omega_cmd = np.sqrt(thrust)
        # thrust[0] = np.sqrt(thrust[0])

        # print(f"ACTION 4: {omega_cmd}")


        return thrust

    def get_control(self):

        # print(f"CURRENT POS: {self.vehicle.pos}")
        # print(f"CURRENT VEL: {self.vehicle.LV}")
        curr = self.vehicle.pos.copy()
        # curr[2] = -curr[2]
        # temp = curr[0]
        # curr[0] = curr[1]
        # curr[1] = temp
        # curr = curr/10
        # cv = self.vehicle.LV.copy()
        # cv[2] = -cv[2]
        # temp = cv[0]
        # cv[0] = cv[1]
        # cv[1] = temp
        # curr[0] = curr[0]
        # self.target_x = 0.0   #(by 10)
        # self.target_y = 0.0
        # self.target_z = 1.0
        target_train_z = 1.0
        curr[0] = curr[0] + self.target_x   
        curr[1] = curr[1] - self.target_y
        curr[2] = curr[2] + (target_train_z - self.target_z)
        vel = (np.linalg.norm(self.vehicle.LV))*10
        print(f"VEL: {vel}")
        # print("goal_local: ", goal_local)
        # eu = [0, 0, 0]
        # r = R.from_euler("zyx", eu)
        r = R.from_quat(self.vehicle.quats)
        # print(self.vehicle.quats)

        ned_to_enu = np.array([[0, 1, 0],
                               [1, 0, 0],
                               [0, 0, -1]])
        
        r_lfu_enu = ned_to_enu @ r.as_matrix() @ ned_to_enu.T    #ned_to_enu = rfu_to_frd  #the correct frame is rfu not lfu

        # print(f"quats: {R.from_matrix(r_lfu_enu).as_quat()}")
        # print(r.as_matrix())
        
        # r_enu = ned_to_enu @ r.as_matrix()

        # print(r_enu)

        angst = R.from_matrix(r_lfu_enu).as_euler("ZYX", degrees=True)

        #angst[0] = 90 + angst[0]
        angst[0] = 0

        rrk = R.from_euler("ZYX", angst, degrees=True).as_matrix()

        # ang[0] = 0
        # ang[0] = 90-ang[0]
        # ang[2] = -ang[2]
        # ang[2] = -ang[2]
        # print(f"EULER ZYX: {ang}")
        #rr = R.from_euler("ZXY", angst, degrees=True)


        # temp = ang[2]
        # ang[2] = ang[1]
        # ang[1] = temp
        # rr = R.from_euler
        # print(f"ANGLE : {angst}")
        # euler = rr.as_matrix()
        # euler = r.as_matrix()

        # euler = np.reshape(euler, (9,), "F")
        # euler = np.reshape(euler, (9,), "F")
        euler = np.reshape(rrk, (9,), "F")
        # print(len(self.vehicle.pos))
        obs = np.concatenate([curr, euler, self.vehicle.LV, self.vehicle.AV, [self.yaw_rate]]).astype(np.float32)

        obs = np.expand_dims(obs, axis=0)
        # obs = np.array([[5.6962529e-03, -6.5228711e-03,  1.0015297e+00,  9.9984783e-01, 1.7444085e-02, -1.6304986e-06, -1.7444085e-02,  9.9984783e-01, 2.7210058e-06,  1.6777152e-06, -2.6921496e-06,  1.0000000e+00, 4.7197806e-05, -8.3191735e-05, -9.5338997e-05, -3.4885236e-06, -1.6702197e-06,  3.5352804e-04,  1.8199615e+00]]).astype(np.float32)
        # print(f"OBSERVATION: {obs}")
        raw_action = self.model.predict(obs)
        # print(f"RAW_ACTION: {raw_action}")

        raw_action = raw_action.numpy()
        #print(raw_action.shape)
        # print(raw_action)

        action = np.zeros((raw_action.shape))

        # # action[:, :, 0] = self.std[0]*raw_action[:, :, 0] + self.mean[0]
        action[:, :, 0] = raw_action[:, :, 0] + 9.81
        # action[0] = action[0] + 9.81
        # # if action[:, :, 0] > self.total_thrust[0]:
        #     # action[:, :, 0] = self.total_thrust[0]

        # # action[:, :, 0] = raw_action[:, :, 0]*self.mean[0]
        action[:, :, 1:3] = raw_action[:, :, 1  :3]*5
        # print(f"RAW_ACTION 2: {action}")
        # action[:,:, 1] = action[:,:, 1]
        #action[:, :, 3] = raw_action[:, :, 3]
        # action[:, :, 3] = -raw_action[:,:, 3]*0.1
        # action[1:4] = action[1:4]*5
        action = action.squeeze(0)
        action = action.squeeze(0)
        # action[3] = -raw_action[:,:, 3]
        # print(action.shape)
        # action[0] = min(action[0], 9.81*1.39)
        # action[0] = action[0]/(9.81*1.39)
       
        thrust = self.get_thrust(action)

        # print(f"Final trhsut {thrust}")

        self.yaw_rate += angst[0]*self.sim_dt

        return thrust
    
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
            return

        if not self.vehicle.offboard:
            self.vehicle.enable_offboard()
            return
        
        # yo = np.array([ 18.8423, 0.00409635, 0.010741, 0.2])

        # yoyo = self.inverse_allocat @ yo

        # print(yoyo)
        
        # if self.i < 300:
        #     # print(f"allo: {self.allocation_matrix}")
        #     # print(f"allo I: {self.inverse_allocat}")
        #     print(self.action_array[self.i, :])
        #     thrust = self.get_control(self.action_array[self.i, :])
        #     self.vehicle.offboard_control()
        #     self.vehicle.publish_ctbr(thrust)
        #     self.i += 1
        thrust = self.get_control()
        # thrust = [1,2,3,4]
        self.vehicle.offboard_control()
        # print(f"THRUST : {thrust}")
        self.vehicle.publish_ctbr(thrust)

        curr_pos = self.vehicle.pos.copy()
        error_x = - self.target_x - curr_pos[0]  # Assuming obs[0] is current x
        error_y = self.target_y - curr_pos[1]  # Assuming obs[1] is current y
        error_z = self.target_z - curr_pos[2]  # Assuming obs[2] is current z
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
                - self.target_x, self.target_y, self.target_z,
                curr_pos[0], curr_pos[1], curr_pos[2],
                error_x, error_y, error_z,
                error_magnitude
            ])

        # print("ACC: ", self.vehicle.LA[2])

        # self.vehicle.set_trajectory([5, 0, -5])

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
    # target = get_input()
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
