  GNU nano 4.8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     data_collection_node.py                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                #!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan, Imu  # Add Imu message type
from obstacle_detector.msg import Obstacles
import tf2_ros
import tf2_geometry_msgs
import csv
import os
import json
from geometry_msgs.msg import TransformStamped

# Directory to save data
save_dir = '/home/aiot/data_collection'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# CSV file to save the data
csv_file = os.path.join(save_dir, 'data.csv')

# Initialize CSV writer
csv_file_handle = open(csv_file, 'w', newline='')
csv_writer = csv.writer(csv_file_handle)
csv_writer.writerow(['timestamp', 'laser_ranges', 'pose_x', 'pose_y', 'pose_theta', 'objects','imu_data'])  # Add 'imu_data' column

# Global variables to store the latest messages
latest_scan = None
latest_obstacles = None
latest_imu = None  # New global variable for IMU data

def laser_callback(data):
    global latest_scan
    latest_scan = data

def obstacles_callback(data):
    global latest_obstacles
    latest_obstacles = data

def imu_callback(data):
    global latest_imu
    latest_imu = data

def get_robot_pose():
    try:
        transform = tf_buffer.lookup_transform('map', 'base_link', rospy.Time(0))
        pose_x = transform.transform.translation.x
        pose_y = transform.transform.translation.y
        pose_theta = transform.transform.rotation.z  # Assuming 2D robot
        return pose_x, pose_y, pose_theta
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        rospy.logerr('Failed to get transform from map to base_link')
        return None, None, None

def collect_data(event):
    if latest_scan and latest_obstacles and latest_imu:  # Check if all data sources are available
        timestamp = rospy.Time.now().to_sec()
        laser_ranges = list(latest_scan.ranges)

        pose_x, pose_y, pose_theta = get_robot_pose()

        object_positions = []
        for circle in latest_obstacles.circles:
            object_positions.append({
                'center_x': circle.center.x,
                'center_y': circle.center.y,
                'true_radius': circle.true_radius
            })

        imu_data = {
            'linear_acceleration': {
                'x': latest_imu.linear_acceleration.x,
                'y': latest_imu.linear_acceleration.y,
                'z': latest_imu.linear_acceleration.z
            },
            'angular_velocity': {
                'x': latest_imu.angular_velocity.x,
                'y': latest_imu.angular_velocity.y,
                'z': latest_imu.angular_velocity.z
            },
            'orientation': {
                'x': latest_imu.orientation.x,
                'y': latest_imu.orientation.y,
                'z': latest_imu.orientation.z,
                'w': latest_imu.orientation.w
            }
        }

        # Write data to CSV file
        csv_writer.writerow([
            timestamp,
            json.dumps(laser_ranges),  # Store laser ranges as JSON string
            pose_x,
            pose_y,
            pose_theta,
            json.dumps(object_positions),  # Store object positions as JSON string
            json.dumps(imu_data)  # Store IMU data as JSON string
        ])

if __name__ == '__main__':
    rospy.init_node('data_collection_node', anonymous=True)

    tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(30))
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rospy.Subscriber('/scan', LaserScan, laser_callback)
    rospy.Subscriber('/object_detector/obstacles', Obstacles, obstacles_callback)
    rospy.Subscriber('/imu', Imu, imu_callback)  # Subscribe to the IMU topic

    # Use a timer to collect data at regular intervals (e.g., every second)
    rospy.Timer(rospy.Duration(1), collect_data)

    rospy.spin()

    csv_file_handle.close()





