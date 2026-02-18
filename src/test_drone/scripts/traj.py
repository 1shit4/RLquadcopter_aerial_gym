import rclpy
from rclpy.node import Node
import numpy as np
import time
from scipy.interpolate import CubicSpline
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from geometry_msgs.msg import Transform, Twist, Vector3, Quaternion
from std_msgs.msg import Header

def create_waypoint(pos, vel, acc):
    traj_point = MultiDOFJointTrajectoryPoint()
    rotation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    transform = Transform(translation=Vector3(x=pos[0], y=pos[1], z=pos[2]), rotation=rotation)
    velocity = Twist(linear=Vector3(x=vel[0], y=vel[1], z=vel[2]))
    acceleration = Twist(linear=Vector3(x=acc[0], y=acc[1], z=acc[2]))
    traj_point.transforms.append(transform)
    traj_point.velocities.append(velocity)
    traj_point.accelerations.append(acceleration)
    return traj_point

def get_hover_point(z_height=1.0):
    return create_waypoint([0.0, 0.0, z_height], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

def get_circular_trajectory_point(radius, speed, time):
    x, y, z = radius * np.cos(speed * time), radius * np.sin(speed * time), 1.5
    vx, vy, vz = -radius * speed * np.sin(speed * time), radius * speed * np.cos(speed * time), 0.0
    ax, ay, az = -radius * speed**2 * np.cos(speed * time), -radius * speed**2 * np.sin(speed * time), 0.0
    return create_waypoint([x, y, z], [vx, vy, vz], [ax, ay, az])

def get_line_trajectory_point(length, duration, time):
    quarter_duration = duration/2
    speed = length/duration
    if time<=quarter_duration:
        x = speed*time
    elif quarter_duration < time <= 2*quarter_duration:
        x = length/2 - (speed*(time - quarter_duration))
    elif 2*quarter_duration < time <= 3*quarter_duration:
        x = -(speed*(time - 2*quarter_duration))
    else:
        x = -length/2 + (speed*(time - 3*quarter_duration))
    # half_duration = duration / 2; speed = length / half_duration
    # if time <= half_duration: x, vx = speed * time, speed
    # else: x, vx = length - (speed * (time - half_duration)), -speed
    y, z = 0.0, 1.5; vx, vy, vz, ax, ay, az = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    # print(vx)
    return create_waypoint([x, y, z], [vx, vy, vz], [ax, ay, az])
def get_figure_eight_trajectory_point(scale, speed, time):
    x, y, z = scale * np.sin(speed * time), scale * np.sin(speed * time)*np.cos(speed*time), 1.5
    vx, vy, vz = -scale * speed * np.sin(speed * time), scale * speed * np.cos(2 * speed * time), 0.0
    ax, ay, az = -scale * speed**2 * np.cos(speed * time), -2 * scale * speed**2 * np.sin(2 * speed * time), 0.0
    return create_waypoint([x, y, z], [vx, vy, vz], [ax, ay, az])

def get_vertical_line_point(height, duration, time, z_start=3.0):
    half_duration = duration / 2; speed = height / half_duration
    if time <= half_duration: z, vz = z_start + (speed * time), speed
    else: z, vz = (z_start + height) - (speed * (time - half_duration)), -speed
    x, y = 0.0, 0.0; vx, vy, ax, ay, az = 0.0, 0.0, 0.0, 0.0, 0.0
    return create_waypoint([x, y, z], [vx, vy, vz], [ax, ay, az])

def get_square_trajectory_point(side_length, speed, time, transition_duration=3.0, corner_rest_time=0.5, z_height=3.0):
    """
    Square trajectory in x-y plane with smooth transition and corner rests.
    
    Args:
        side_length: Side length of square (e.g., 3.0 for 3x3)
        speed: Movement speed along edges (m/s)
        time: Current time
        transition_duration: Time to reach square start from center (seconds)
        corner_rest_time: Time to rest at each corner (seconds)
        z_height: Height of square trajectory
    """
    half_side = side_length / 2.0
    
    # Square corners (starting from top-right, going counter-clockwise)
    corners = [
        (half_side, half_side),      # Corner 0: Top-right (start)
        (-half_side, half_side),     # Corner 1: Top-left
        (-half_side, -half_side),    # Corner 2: Bottom-left
        (half_side, -half_side),     # Corner 3: Bottom-right
    ]
    
    # Phase 1: Smooth transition from center (0,0) to first corner
    if time < transition_duration:
        t_norm = time / transition_duration
        # Cubic easing for smooth acceleration/deceleration
        s = 3 * t_norm**2 - 2 * t_norm**3
        
        x = corners[0][0] * s
        y = corners[0][1] * s
        z = z_height
        
        return create_waypoint([x, y, z], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    
    # Phase 2: Square motion with corner rests
    adjusted_time = time - transition_duration
    
    # Time to traverse one edge
    edge_time = side_length / speed
    
    # Total time for one segment (edge + corner rest)
    segment_time = edge_time + corner_rest_time
    
    # Total time for complete square loop
    loop_time = 4 * segment_time
    
    # Loop the time
    t = adjusted_time % loop_time
    
    # Determine current segment (0-3 for four sides)
    segment_index = int(t / segment_time)
    time_in_segment = t % segment_time
    
    # Current corner and next corner
    current_corner = corners[segment_index]
    next_corner = corners[(segment_index + 1) % 4]
    
    # Check if we're in corner rest period or moving along edge
    if time_in_segment < corner_rest_time:
        # Resting at corner
        x, y = current_corner
    else:
        # Moving along edge
        edge_progress = (time_in_segment - corner_rest_time) / edge_time
        
        # Linear interpolation between current and next corner
        x = current_corner[0] + (next_corner[0] - current_corner[0]) * edge_progress
        y = current_corner[1] + (next_corner[1] - current_corner[1]) * edge_progress
    
    z = z_height
    
    return create_waypoint([x, y, z], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

def get_looping_helical_point(radius, angular_speed, vertical_speed, height, time, z_start=3.0):
    x, y = radius * np.cos(angular_speed * time), radius * np.sin(angular_speed * time)
    vx, vy = -radius * angular_speed * np.sin(angular_speed * time), radius * angular_speed * np.cos(angular_speed * time)
    ax, ay = -radius * angular_speed**2 * np.cos(angular_speed * time), -radius * angular_speed**2 * np.sin(angular_speed * time)
    half_period = height / vertical_speed; full_period = 2 * half_period; cycle_time = time % full_period
    if cycle_time <= half_period: z, vz = z_start + (vertical_speed * cycle_time), vertical_speed
    else: z, vz = (z_start + height) - (vertical_speed * (cycle_time - half_period)), -vertical_speed
    az = 0.0
    return create_waypoint([x, y, z], [vx, vy, vz], [ax, ay, az])

def get_smooth_circle_trajectory(radius, angular_speed, time, transition_duration, z_height=1.5):
    """
    Smooth transition from (0,0) to circle start, then circular motion.
    Only sends position setpoints (velocity and acceleration set to zero).
    """
    circle_start_x = radius
    circle_start_y = 0.0
    
    if time < transition_duration:
        # Smooth transition phase using cubic interpolation
        t_norm = time / transition_duration
        # Cubic easing: s = 3t² - 2t³
        s = 3 * t_norm**2 - 2 * t_norm**3
        
        # Position: interpolate from (0,0) to (radius, 0)
        x = circle_start_x * s
        y = circle_start_y * s
        z = z_height
        
    else:
        # Normal circular motion
        adjusted_time = time - transition_duration
        theta = angular_speed * adjusted_time
        
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = z_height
    
    # Always zero velocity and acceleration
    return create_waypoint([x, y, z], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])

class SmoothRandomTrajectory(Node):
    """
    ROS 2 node that continuously generates smooth random 3D waypoints
    and interpolates a trajectory between them with varying velocities.
    """

    def __init__(self):
        super().__init__('smooth_random_trajectory')

        self.trajectory_pub = self.create_publisher(MultiDOFJointTrajectory, '/command/trajectory', 10)

        # Bounds for position and velocity

        self.prev_point = np.array([0.0, 0.0, 4.0])  # Start position
        self.start_time_ = self.get_clock().now().nanoseconds / 1e9

        self.timer = self.create_timer(0.1, self.publish_next_trajectory)  # 10Hz loop

        self.get_logger().info("Smooth random trajectory node started.")

        self.TRAJECTORY_TYPE = "CIRCLE"  # Options: HOVER, CIRCLE, LINE, FIGURE_EIGHT, VERTICAL_LINE, HELIX_LOOP, SQUARE
        self.LINE_HEIGHT, self.LINE_DURATION = 10.0, 15.0
        self.RADIUS, self.ANGULAR_SPEED = 1.5, 0.15
        self.TRANSITION_DURATION = 4.5  # seconds to reach circle start
        self.circle_z = 1.5
        self.VERTICAL_SPEED, self.MAX_HEIGHT = 0.25, 5.0
        self.LINE_LENGTH = 3.0
        self.SCALE, self.SPEED = 1.5, 0.1
        self.SQUARE_SIDE = 3.0           # 3x3 square
        self.SQUARE_SPEED = 0.1          # m/s along edges
        self.SQUARE_CORNER_REST = 1   # 0.5s rest at corners
        self.SQUARE_TRANSITION = 4     # 3s to reach start
        self.SQUARE_HEIGHT = 1.5         # Height in z


    def publish_next_trajectory(self):
        """Generate and publish a short smooth trajectory to a new random point."""
        current_time = (self.get_clock().now().nanoseconds / 1e9) - self.start_time_
        
        # THE FIX #2: Add modulo operator for looping trajectories.
        if self.TRAJECTORY_TYPE == "HOVER":
            trajectory_point = get_hover_point()
        elif self.TRAJECTORY_TYPE == "CIRCLE":
            trajectory_point = get_smooth_circle_trajectory(self.RADIUS, self.ANGULAR_SPEED, current_time, self.TRANSITION_DURATION, self.circle_z)
            # trajectory_point = get_circular_trajectory_point(self.RADIUS, self.ANGULAR_SPEED, current_time)
        elif self.TRAJECTORY_TYPE == "LINE":
            looped_time = current_time % self.LINE_DURATION
            trajectory_point = get_line_trajectory_point(self.LINE_LENGTH, self.LINE_DURATION, looped_time)
        elif self.TRAJECTORY_TYPE == "FIGURE_EIGHT":
            trajectory_point = get_figure_eight_trajectory_point(self.SCALE, self.SPEED, current_time)
        elif self.TRAJECTORY_TYPE == "VERTICAL_LINE":
            looped_time = current_time % (self.LINE_DURATION*2)
            trajectory_point = get_vertical_line_point(self.LINE_HEIGHT, self.LINE_DURATION, looped_time)
        elif self.TRAJECTORY_TYPE == "HELIX_LOOP":
            trajectory_point = get_looping_helical_point(self.RADIUS, self.ANGULAR_SPEED, self.VERTICAL_SPEED, self.MAX_HEIGHT, current_time)
        elif self.TRAJECTORY_TYPE == "SQUARE":
            trajectory_point = get_square_trajectory_point(
                self.SQUARE_SIDE, self.SQUARE_SPEED, current_time,
                self.SQUARE_TRANSITION, self.SQUARE_CORNER_REST, self.SQUARE_HEIGHT
            )
        else:
            self.get_logger().error(f"Invalid TRAJECTORY_TYPE: {self.TRAJECTORY_TYPE}")
            trajectory_point = get_hover_point() # Default to hover for safety
            
            
        header = Header(stamp=self.get_clock().now().to_msg(), frame_id='world')
        trajectory_msg = MultiDOFJointTrajectory(header=header, points=[trajectory_point])

        self.trajectory_pub.publish(trajectory_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SmoothRandomTrajectory()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()