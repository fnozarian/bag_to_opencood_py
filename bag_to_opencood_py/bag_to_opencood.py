import os
import subprocess
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from transforms3d import euler
import open3d as o3d
from sensor_msgs_py import point_cloud2 as pc2

"""
This node subscribes to a relative pose and lidar topics from two vehicles and synchronizes the messages using Approximate Time Synchronizer.
The synchronized relative pose messages are saved as YAML files and the lidar point clouds are saved as PCD files in OpenCood format.
Run this node and play the merged pose and lidar bag file using the following command:
```ros2 bag play /workspace/merged_bag/merged_bag_0.db3 --clock -p -r 0.2```
"""
class SyncAndWriteNode(Node):
    def __init__(self):
        super().__init__('sync_and_write_node')
        
        self.get_logger().info('Sync and Write Node has started!')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.declare_parameter('output_directory', '/workspace/v2v_dataset/2024-03-19-14-18-43/')
        self.declare_parameter('input_bag', '/workspace/htw_rosbag/V2X/v2x_bag.db3')
        self.declare_parameter('lidar_topic1', '/ouster/points/kia1')
        self.declare_parameter('pose_topic1', '/pose/kia1')
        self.declare_parameter('lidar_topic2', '/ouster/points/kia2')
        self.declare_parameter('pose_topic2', '/pose/kia2')
        self.output_directory = self.get_parameter('output_directory').get_parameter_value().string_value
        self.input_bag = self.get_parameter('input_bag').get_parameter_value().string_value
        self.lidar_topic1 = self.get_parameter('lidar_topic1').get_parameter_value().string_value
        self.pose_topic1 = self.get_parameter('pose_topic1').get_parameter_value().string_value
        self.lidar_topic2 = self.get_parameter('lidar_topic2').get_parameter_value().string_value
        self.pose_topic2 = self.get_parameter('pose_topic2').get_parameter_value().string_value

        # Create directories if they don't exist
        os.makedirs(os.path.join(self.output_directory, '0'), exist_ok=True)
        os.makedirs(os.path.join(self.output_directory, '1'), exist_ok=True)

        # Create subscribers
        self.pose_sub1 = Subscriber(self, PoseStamped, self.pose_topic1)
        self.lidar_sub1 = Subscriber(self, PointCloud2, self.lidar_topic1)
        self.pose_sub2 = Subscriber(self, PoseStamped, self.pose_topic2)
        self.lidar_sub2 = Subscriber(self, PointCloud2, self.lidar_topic2)

        # Set up the synchronization policy (Approximate Time Synchronizer)
        self.ts = ApproximateTimeSynchronizer([self.pose_sub1, self.lidar_sub1, self.pose_sub2, self.lidar_sub2], queue_size=50, slop=1)
        self.ts.registerCallback(self.sync_callback)

        self.frame_count = 0

        self.play_bag_file(self.input_bag)

    def play_bag_file(self, bag_path):
        command = ['ros2', 'bag', 'play', bag_path, '--clock', '-d', '2', '-r', '0.2']
        self.bag_process = subprocess.Popen(command)
        self.get_logger().info(f"Playing bag file: {bag_path}")

    # def clock_callback(self, msg):
    #     self.get_logger().info(f"Clock: {msg.clock}")    
    
    def save_pcd(self, pc_msg, file_name):
        # Extract points and intensity from ROS point cloud
        points = []
        intensities = []

        for point in pc2.read_points(pc_msg, skip_nans=True, field_names=("x", "y", "z", "intensity")):
            points.append([point[0], point[1], point[2]])
            intensities.append(point[3])

        points = np.array(points)
        intensities = np.array(intensities)

        # Flip y-coordinates to transform to left-hand coordinate system (OpenCOOD)
        # points[:, 1] = -points[:, 1]

        # Normalize intensity to [0, 1]
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min())
        colors = np.stack([intensities, intensities, intensities], axis=-1)  # Use intensity as gray color

        o3d_cloud = o3d.geometry.PointCloud()
        o3d_cloud.points = o3d.utility.Vector3dVector(points)
        o3d_cloud.colors = o3d.utility.Vector3dVector(colors)

        o3d.io.write_point_cloud(file_name, o3d_cloud)

        return o3d_cloud
    
    def save_yaml(self, pose_msg, file_name):
        # Extract timestamp from the pose message
        timestamp = f"{pose_msg.header.stamp.sec}.{pose_msg.header.stamp.nanosec:09d}"
        # Convert pose message to a 4x4 transformation matrix
        matrix = self.pose_to_matrix(pose_msg)

        # Create YAML dictionary
        yaml_dict = {
            'ego_speed': 0,
            'gps': [],
            'lidar_pose': matrix,
            'true_ego_pos': matrix,
            'vehicles': [],
            'timestamp': timestamp
        }

        # Write to YAML file
        with open(file_name, 'w') as yaml_file:
            yaml.dump(yaml_dict, yaml_file,  default_flow_style=False)
        
        return yaml_dict

    def sync_callback(self, pose_msg1, lidar_msg1, pose_msg2, lidar_msg2):

        lidar_time1 = lidar_msg1.header.stamp.sec + lidar_msg1.header.stamp.nanosec * 1e-9
        pose_time1 = pose_msg1.header.stamp.sec + pose_msg1.header.stamp.nanosec * 1e-9
        lidar_time2 = lidar_msg2.header.stamp.sec + lidar_msg2.header.stamp.nanosec * 1e-9
        pose_time2 = pose_msg2.header.stamp.sec + pose_msg2.header.stamp.nanosec * 1e-9

        times = np.array([lidar_time1, pose_time1, lidar_time2, pose_time2])
        max_time_diff = np.max(times) - np.min(times)
        self.get_logger().info(f"Pose 1 time {pose_time1}")
        self.get_logger().info(f"Lidar 1 time {lidar_time1}")
        self.get_logger().info(f"Pose 2 time {pose_time2}")
        self.get_logger().info(f"Lidar 2 time {lidar_time2}")
        self.get_logger().info(f"Frame {self.frame_count} - Max time diff {max_time_diff}")
        self.get_logger().info(f"===============================================")

        self.save_yaml(pose_msg1, os.path.join(self.output_directory, '0', f"{self.frame_count:06d}.yaml"))
        self.save_yaml(pose_msg2, os.path.join(self.output_directory, '1', f"{self.frame_count:06d}.yaml"))
        
        self.save_pcd(lidar_msg1, os.path.join(self.output_directory, '0', f"{self.frame_count:06d}.pcd"))
        self.save_pcd(lidar_msg2, os.path.join(self.output_directory, '1', f"{self.frame_count:06d}.pcd"))

        self.get_logger().info(f"Frame {self.frame_count} saved.")

        self.frame_count += 1
    
    def __del__(self):
        if hasattr(self, 'bag_process'):
            self.bag_process.terminate()
            self.get_logger().info("Bag file playback terminated")

    def pose_to_matrix(self, pose_msg):
        """Converts a PoseStamped message to a 4x4 transformation matrix."""
        position = [pose_msg.pose.position.x, pose_msg.pose.position.y, pose_msg.pose.position.z]
        quaternion = [pose_msg.pose.orientation.w, pose_msg.pose.orientation.x, pose_msg.pose.orientation.y, pose_msg.pose.orientation.z]
        rotation_matrix = euler.quat2mat(quaternion)

        # Create the 4x4 transformation matrix
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_matrix
        matrix[:3, 3] = position
        return matrix

def main(args=None):
    rclpy.init(args=args)
    node = SyncAndWriteNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
