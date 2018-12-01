import rospy
import numpy as np
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
import math
from math import hypot, atan2, pi
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion

class Env():
    def __init__(self):
        self.goal_x = 0 # find cor of the goal and insert
        self.goal_y = -5 # find cor of the goal and insert
        self.heading = 0
        self.position = Pose()
        self.sub_odom = rospy.Subscriber('obom',Odometry,self.get_Theta)
        self.get_goalbox = False
        self.goal=[]
        self.run_theta = pi/4 # chk...

    def getGoalDistance(self):
        GoalDistance = round(hypot(self.goal_x-self.position.x,self.goal_y-self.position.y),2)
        return GoalDistance

    def get_Theta(self,odom):
        self.position = lidar_data.pose.position
        goal_angle = atan2(self.goal_y-self.position.y,self.goal_x-self.position.x)
        
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        
        #goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle# - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    '''
    def scancallback(self,msg):
    	print msg.ranges

    def posecallback(self,msg):
    	print msg.pose.position
    '''

    def get_State(self, lidar_data):
        scan_range=[]
        heading = self.heading
        min_range = 0.13 # consider car size
        done =False

        for i in range(len(lidar_data.ranges)):
            if lidar_data.ranges[i] == float('Inf'):
                scan_range.append(5)
            elif np.isnan(lidar_data.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(lidar_data)
        obstacle_min_range = round(min(scan_range),2)
        obstacle_angle = np.argmin(scan_range)

        if min_range > min(scan_range) > 0:
            done = True
        
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.2: #close to goal
            self.get_goalbox = True

        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done

    def set_Reward(self, state, done, action):

        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]

        if action == 3:
            angle = -pi / 4 + heading + (self.run_theta) + pi / 2
            yaw_reward = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
        if action == 4:
            angle = -pi / 4 + heading + (self.run_theta) + pi / 2
            yaw_reward = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
    
        distance_rate = 2**(current_distance / self.getGoalDistance())

        if obstacle_min_range < 0.5:
            ob_reward = -5
        else:
            ob_reward = 0
        reward = ((round(yaw_reward*5, 2)) * distance_rate) + ob_reward

        if done:
            reward -= 500

        if self.get_goalbox:
            reward += 1000
            self.get_goalbox = False

        return reward

    def step(self, action):

        data = None
        
        while data is None:
            try:
                data = rospy.wait_for_message('msg',LaserScan,timeout=5)
            except:
                pass

        state, done = self.get_State(data)
        reward = self.set_Reward(state, done, action)

        if self.get_goalbox is True:
            done = True

        return np.asarray(state), reward, done

    def reset(self):
        
        data = None
        
        while data is None:
            try:
                data = rospy.wait_for_message('msg',LaserScan,timeout=5)
            except:
                pass
        
        state, _ = self.get_State(data)

        return np.asarray(state)