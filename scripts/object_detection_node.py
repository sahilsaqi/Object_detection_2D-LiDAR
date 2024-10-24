  GNU nano 4.8                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    object_detection_node.py                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               #!/usr/bin/env python3

import rospy
import math
import numpy as np
from copy import deepcopy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PointStamped, Point
from obstacle_detector.msg import Obstacles, CircleObstacle
from tensorflow.keras.models import load_model
import joblib

# Parameters
thres_shortest = 5.0  # units: m
thres_obj_gap = 0.40  # units: m
obs_max_size = 30     # units : obstacle laser scan points
obs_min_size = 15
alpha = 0.1
prev_theta = 0

# Load the model and scaler globally
model = None
scaler = None

def load_model_and_scaler():
    global model, scaler
    model = load_model('models/my_model.keras')
    scaler = joblib.load('models/scaler.pkl')

# Publishers
robot_id = rospy.get_param('robot_id', '')
pub_obj = rospy.Publisher(robot_id + "/object_detector/clustered_closest_obj", LaserScan, queue_size=10)
pub_point = rospy.Publisher(robot_id + "/object_detector/obstacles", Obstacles, queue_size=10)

def deg2rad(data):
    return math.radians(data)

def dataCounter(data):
    j = 0
    for i in range(len(data.ranges)):
        if data.ranges[i] != 0.0:
            j += 1
    return j

def findShortest(data):
    shortest = shortest_idx = 0
    shortest_flag = False
    flag = True
    for i in range(len(data.ranges)):
        if data.ranges[i] == 0:
            continue
        elif data.ranges[i] < thres_shortest and flag:
            shortest = data.ranges[i]
            shortest_idx = i
            shortest_flag = True
            flag = False
        elif data.ranges[i] < shortest and not flag:
            shortest = data.ranges[i]
            shortest_idx = i

    return shortest_flag, shortest, shortest_idx

def convert_objects_to_features(objects):
    features = [0] * 14  # Adjust the size according to your feature vector
    for i, obj in enumerate(objects):
        if i < 4:  # Assuming up to 4 objects
            features[i * 3] = obj.center.x
            features[i * 3 + 1] = obj.center.y
            features[i * 3 + 2] = obj.true_radius

    imu_data = [0] * 8  # Replace with actual IMU data if available
    features[5:13] = imu_data

    return np.array(features).reshape(1, -1)

def preprocess_and_predict(features):
    global model, scaler
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    return predictions

def LaserHandler(data):
    idx = []
    obs_msg = LaserScan()
    obs_point_msg = Obstacles()
    filtered_msg = deepcopy(data)
    filtered_msg.ranges = list(filtered_msg.ranges)
    obs_msg.ranges = [0.0] * len(data.ranges)

    while True:
        shortest_flag, shortest, shortest_idx = findShortest(filtered_msg)

        if not shortest_flag:
            break

        left_idx = right_idx = shortest_idx
        left_flag = right_flag = True
        l_tmp = r_tmp = shortest

        idx.append(shortest_idx)
        for i in range(len(filtered_msg.ranges)):
            if left_idx >= len(filtered_msg.ranges) - 1:
                left_flag = False
            else:
                if filtered_msg.ranges[left_idx + 1] == 0.0 and left_flag:
                    left_idx += 1
                elif abs(l_tmp - filtered_msg.ranges[left_idx + 1]) < thres_obj_gap and left_flag:
                    left_idx += 1
                    l_tmp = filtered_msg.ranges[left_idx]
                    idx.append(left_idx)
                    obs_msg.ranges[left_idx] = filtered_msg.ranges[left_idx]
                    filtered_msg.ranges[left_idx] = 0.0
                else:
                    left_flag = False

            if right_idx <= 0:
                right_flag = False
            else:
                if filtered_msg.ranges[right_idx - 1] == 0.0 and right_flag:
                    right_idx -= 1
                elif abs(r_tmp - filtered_msg.ranges[right_idx - 1]) < thres_obj_gap and right_flag:
                    right_idx -= 1
                    r_tmp = filtered_msg.ranges[right_idx]
                    idx.append(right_idx)
                    obs_msg.ranges[right_idx] = filtered_msg.ranges[right_idx]
                    filtered_msg.ranges[right_idx] = 0.0
                else:
                    right_flag = False

            if not left_flag and not right_flag:
                filtered_msg.ranges[shortest_idx] = 0.0
                idx.sort(reverse=True)
                break

        if len(idx) < obs_min_size:
            for i in range(len(idx)):
                obs_msg.ranges[idx[i]] = 0.0
        else:
            theta = int(((idx[0] + idx[-1]) / 2) * 86 / len(data.ranges) - 149)
            distance = data.ranges[idx[int(len(idx) / 2)]]
            x = distance * math.cos(deg2rad(theta))
            y = distance * math.sin(deg2rad(theta))
            print("# of Laser Points : ", len(idx), " : ", idx)
            print("(x, y) = (", x, ",", y, ")")

            tmp = CircleObstacle()
            tmp.center.x = x
            tmp.center.y = y
            tmp.center.z = 0
            tmp.true_radius = 0.01 * len(idx)  # Heuristic value
            obs_point_msg.circles.append(tmp)

        idx = []

    pub_obj.publish(obs_msg)
    obs_point_msg.header.frame_id = data.header.frame_id
    pub_point.publish(obs_point_msg)

    # Use ML model for prediction on processed data
    features_from_objects = convert_objects_to_features(obs_point_msg.circles)
    predictions = preprocess_and_predict(features_from_objects)

    # Log the predictions
    rospy.loginfo(f'Predictions: {predictions}')

def main():
    rospy.init_node('multi_object_detector_node', anonymous=True)
    load_model_and_scaler()
    rospy.Subscriber(robot_id + "/scan", LaserScan, LaserHandler)
    rospy.spin()

if __name__ == '__main__':
    main()




