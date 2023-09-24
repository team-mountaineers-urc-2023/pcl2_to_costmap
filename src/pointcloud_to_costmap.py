#!/usr/bin/env python3

import numpy as np
from matplotlib.path import Path
from scipy.interpolate import NearestNDInterpolator
from scipy.stats import binned_statistic_2d

import rospy
from sensor_msgs import point_cloud2 as PCL2

from geometry_msgs.msg import Pose, Quaternion, Vector3
from std_msgs.msg import Header
from nav_msgs.msg import MapMetaData, OccupancyGrid
from sensor_msgs.msg import PointCloud2

import numpy as np
import math

import tf2_ros
import tf2_py as tf2
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

### main #####################################################################

def main():
	PointCloudToCostamp().loop()

class PointCloudToCostamp:
	def __init__(self) -> None:

		rospy.init_node('pointcloud_to_costmap')

		### local variables ##################################################

		self.seq = 0

		self.width = rospy.get_param("~width")
		self.height = rospy.get_param("~height")
		self.resolution = rospy.get_param("~resolution")
		self.rows = round(self.width / self.resolution)
		self.cols = round(self.height / self.resolution)
		self.lidar_height = rospy.get_param("~lidar_height")
		self.driveable_slope = rospy.get_param("~driveable_slope")
		self.frame_id = rospy.get_param("~frame_id")
		self.use_nearest_neighbor_approximation = rospy.get_param("~use_nearest_neighbor_approximation")

		self.clear_zone = Path(rospy.get_param("~clear_zone"))
		self.max_point_height = rospy.get_param("~max_point_height")

		### connect to ROS ###################################################

		pointcloud_topic = rospy.get_param("~pointcloud_topic")
		costmap_topic = rospy.get_param("~costmap_topic")

		self.pointcloud_sub = rospy.Subscriber(pointcloud_topic, PointCloud2, self.pointcloud_cb)
		self.costmap_pub = rospy.Publisher(costmap_topic, OccupancyGrid, queue_size=1)

		self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(12))
		self.tl = tf2_ros.TransformListener(self.tf_buffer)

		self.do_tf = rospy.get_param("~transform_lidar_to_earth")
		# self.target_frame = rospy.get_param("~lidar_tf_target_frame")
		# self.source_frame = rospy.get_param("~lidar_tf_source_frame")
		# self.lookup_time = rospy.get_param("~lidar_tf_lookup_time")

		### end init #########################################################

	### callbacks ############################################################

	def pointcloud_cb(self, pcl: PointCloud2):
		if self.do_tf:
			# Need to get the transform from the lidar_link to the link that is flat (not rotated with the rover)
			trans = self.tf_buffer.lookup_transform("map", "lidar_link", rospy.Time(),
														rospy.Duration(0.5))
			
			initial_orientation = trans.transform.rotation
			initial_euler = self.euler_from_quaternion(initial_orientation)
			new_euler = (initial_euler[0], initial_euler[1], 0)
			new_quaternion = self.get_quaternion_from_euler(new_euler)

			new_position = Vector3()


			trans.transform.rotation = new_quaternion
			trans.transform.translation = new_position

			# Transform the cloud so that the point heights are correct
			tf_pcl = do_transform_cloud(pcl, trans)
		else:
			tf_pcl = pcl

		pcl_generator = PCL2.read_points(tf_pcl, skip_nans=True, field_names=('x', 'y', 'z'))
		points = list(pcl_generator)
		xyz = np.stack(points)

		# remove points that are too tall to be anything but noise
		# ok_height = xyz[:, 2] < self.max_point_height
		# xyz_ok_height = xyz[ok_height]
		xyz_ok_height = xyz # Don't denoise, why Shubh?

		# remove points inside the robot's 2D polygon
		xy = xyz_ok_height[:, :2]
		inside = self.clear_zone.contains_points(xy)
		outside = np.invert(inside)

		# print("outside: ", len(outside))
		# print("xyz: ", len(xyz))
		# print("ok_height: ", len(xyz_ok_height))

		xyz_filtered = xyz[outside]

		# breakdown data before binning
		# x, y, z = xyz_filtered.transpose()
		x, y, z = xyz_filtered.transpose()

		# compute height map by binning pointcloud
		height_map = binned_statistic_2d(
			y, x, z,  # not sure why x, and y need to be reversed here - might indicate an error elsewhere
			statistic='max',
			bins=(self.rows, self.cols),
			range=[
				[-self.width // 2, self.width // 2],
				[-self.height // 2, self.height // 2]
			],
			expand_binnumbers = False
		).statistic

		# fill in cells with no height
		if self.use_nearest_neighbor_approximation:
			mask = np.where(~np.isnan(height_map))
			interpolator = NearestNDInterpolator(np.transpose(mask), height_map[mask])
			filled_height_map = interpolator(*np.indices(height_map.shape))
		else:
			filled_height_map = np.nan_to_num(height_map, copy=True, nan=-self.lidar_height, posinf=-self.lidar_height, neginf=-self.lidar_height)

		# compute gradient across height map
		x_gradients, y_gradients = np.gradient(filled_height_map, self.resolution)
		combined_gradients = np.hypot(x_gradients, y_gradients)
		occupancy = [round(100 * g / self.driveable_slope) if g < self.driveable_slope else 100 for g in combined_gradients.flatten()]

		

		# construct header
		header = Header()
		header.seq = self.seq
		header.stamp = rospy.Time.now()
		header.frame_id = self.frame_id
		self.seq +=1

		# construct origin
		origin = Pose()
		origin.position.x = -self.width // 2
		origin.position.y = -self.height // 2
		origin.position.z = -self.lidar_height

		# construct info
		info = MapMetaData()
		info.map_load_time = pcl.header.stamp
		info.resolution = self.resolution
		info.width = self.rows
		info.height = self.cols
		info.origin = origin

		# construct and publish occupancy grid
		costmap = OccupancyGrid()
		costmap.header = header
		costmap.info = info
		costmap.data = occupancy
		# costmap.data = [int(x*50) for x  in filled_height_map.flatten()]
		self.costmap_pub.publish(costmap)

	def get_quaternion_from_euler(self, euler):
		"""
		Convert an Euler angle to a quaternion.
		
		Input
			:param roll: The roll (rotation around x-axis) angle in radians.
			:param pitch: The pitch (rotation around y-axis) angle in radians.
			:param yaw: The yaw (rotation around z-axis) angle in radians.
		
		Output
			:return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
		"""
		roll = euler[0]
		pitch = euler[1]
		yaw = euler[2]

		quat = Quaternion()

		quat.x = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
		quat.y = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
		quat.z = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
		quat.w = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
		
		return quat

	def euler_from_quaternion(self, quat):
		"""
		Convert a quaternion into euler angles (roll, pitch, yaw)
		roll is rotation around x in radians (counterclockwise)
		pitch is rotation around y in radians (counterclockwise)
		yaw is rotation around z in radians (counterclockwise)
		"""
		x = quat.x
		y = quat.y
		z = quat.z
		w = quat.w

		t0 = +2.0 * (w * x + y * z)
		t1 = +1.0 - 2.0 * (x * x + y * y)
		roll_x = math.atan2(t0, t1)
	
		t2 = +2.0 * (w * y - z * x)
		t2 = +1.0 if t2 > +1.0 else t2
		t2 = -1.0 if t2 < -1.0 else t2
		pitch_y = math.asin(t2)
	
		t3 = +2.0 * (w * z + x * y)
		t4 = +1.0 - 2.0 * (y * y + z * z)
		yaw_z = math.atan2(t3, t4)
	
		return (roll_x, pitch_y, yaw_z) # in radians


	### loop #################################################################

	def loop(self):
		rospy.spin()


if __name__ == '__main__':
	main()
