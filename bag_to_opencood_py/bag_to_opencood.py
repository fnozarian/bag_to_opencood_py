import os
import yaml
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import PointCloud2
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.time import Time
from transforms3d import euler
import open3d as o3d
from sensor_msgs_py import point_cloud2 as pc2

"""
This node subscribes to a relative pose and lidar topics and synchronizes the messages using Approximate Time Synchronizer.
The synchronized messages are saved to a YAML file and the lidar point cloud is saved to a PCD file in OpenCood format.
Run this node and play the merged pose and lidar bag file using the following command:
```ros2 bag play /workspace/merged_bag/merged_bag_0.db3 --clock -p -r 0.2```
"""
class SyncAndWriteNode(Node):
    def __init__(self):
        super().__init__('sync_and_write_node')
        self.get_logger().info('Sync and Write Node has started!')
    
        self.declare_parameter('output_directory', '/workspace/synchronized_poses/2024-03-19-14-18-43/0')
        self.declare_parameter('lidar_topic', '/ouster/points')
        self.declare_parameter('pose_topic', '/pose')
        self.output_directory = self.get_parameter('output_directory').get_parameter_value().string_value
        self.lidar_topic = self.get_parameter('lidar_topic').get_parameter_value().string_value
        self.pose_topic = self.get_parameter('pose_topic').get_parameter_value().string_value

        # Create directory if it doesn't exist
        os.makedirs(self.output_directory, exist_ok=True)

        # Create subscribers
        self.pose_sub = Subscriber(self, PoseStamped, self.pose_topic)
        self.lidar_sub = Subscriber(self, PointCloud2, self.lidar_topic)

        # Set up the synchronization policy (Approximate Time Synchronizer)
        self.ts = ApproximateTimeSynchronizer([self.pose_sub, self.lidar_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)
        
        self.frame_count = 0
    
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

    def sync_callback(self, pose_msg, lidar_msg):

        lidar_time = lidar_msg.header.stamp.sec + lidar_msg.header.stamp.nanosec * 1e-9
        pose_time = pose_msg.header.stamp.sec + pose_msg.header.stamp.nanosec * 1e-9
        times = np.array([lidar_time, pose_time])
        max_time_diff = np.max(times) - np.min(times)
        print(f"Frame {self.frame_count} - Max time diff {max_time_diff}")

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

        # Generate the file name based on the timestamp
        file_name = os.path.join(self.output_directory, f"{self.frame_count:06d}.yaml")

        # Write to YAML file
        with open(file_name, 'w') as yaml_file:
            yaml.dump(yaml_dict, yaml_file,  default_flow_style=False)

        # Save the point cloud to a PCD file
        pcd_file_name = os.path.join(self.output_directory, f"{self.frame_count:06d}.pcd")
        self.save_pcd(lidar_msg, pcd_file_name)

        self.get_logger().info(f'Wrote synchronized pose to {file_name}')
        self.frame_count += 1
    

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
