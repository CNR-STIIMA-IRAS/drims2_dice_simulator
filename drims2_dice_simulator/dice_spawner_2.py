import math
import os
import random
import numpy as np
import trimesh

from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Point, Vector3
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster, Buffer, TransformListener
from tf_transformations import quaternion_from_euler, quaternion_multiply
from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject, PlanningScene as PlanningSceneMsg
from shape_msgs.msg import SolidPrimitive, MeshTriangle, Mesh
from std_msgs.msg import Int16

from drims2_msgs.srv import DiceIdentification

class DiceSpawner(Node):
    def __init__(self):
        super().__init__('dice_spawner_node')

        self.declare_parameter("face_up", 0)
        face = self.get_parameter("face_up").get_parameter_value().integer_value
        self.face = face if 1 <= face <= 6 else random.randint(1, 6)

        self.dice_name = "dice"
        self.declare_parameter("dice_size", 0.02)
        self.dice_size = self.get_parameter("dice_size").get_parameter_value().double_value

        default_position = [0.5, 0.0, 0.85]
        self.declare_parameter("position", default_position)
        pos_param = self.get_parameter("position").get_parameter_value().double_array_value
        self.position = Point(x=pos_param[0], y=pos_param[1], z=pos_param[2])

        package_path = get_package_share_directory('drims2_dice_simulator')
        self.dice_mesh_path = os.path.join(package_path, 'urdf', 'Die-OBJ.obj')

        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.dice_face_publisher_ = self.create_publisher(Int16, '/dice_face', 10)
        self.srv = self.create_service(DiceIdentification, '/dice_identification', self.get_dice_state_callback)

        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /apply_planning_scene service...')

        self.scene_subscriber = self.create_subscription(
            PlanningSceneMsg,
            '/monitored_planning_scene',
            self.monitored_scene_callback,
            1
        )
        self.latest_scene = None

        self.face_normals = {
            1: np.array([0, 1, 0]),
            2: np.array([1, 0, 0]),
            3: np.array([0, 0, -1]),
            4: np.array([0, 0, 1]),
            5: np.array([-1, 0, 0]),
            6: np.array([0, -1, 0]),
        }

        self.dice_spawned = False

        self.publish_base_tf()
        self.dynamic_tf_timer = self.create_timer(0.05, self.publish_dynamic_transforms)
        self.scene_tracking_timer = self.create_timer(0.05, self.update_dice_tf_from_scene)
        self.spawn_timer = self.create_timer(2.0, self.spawn_dice_with_mesh_once)

        self.dice_face_publisher_.publish(Int16(data=self.face))

    def publish_base_tf(self):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "world"
        tf.child_frame_id = "dice_base_tf"
        tf.transform.translation = Vector3(
            x=self.position.x,
            y=self.position.y,
            z=self.position.z
        )
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform([tf])

        q = self.get_orientation_for_face(self.face)
        tf1 = TransformStamped()
        tf1.header.stamp = self.get_clock().now().to_msg()
        tf1.header.frame_id = "dice_base_tf"
        tf1.child_frame_id = "dice_rotated_tf"
        tf1.transform.translation.x = 0.0
        tf1.transform.translation.y = 0.0
        tf1.transform.translation.z = 0.0
        tf1.transform.rotation.x = q[0]
        tf1.transform.rotation.y = q[1]
        tf1.transform.rotation.z = q[2]
        tf1.transform.rotation.w = q[3]
        self.static_tf_broadcaster.sendTransform([tf1])

    def monitored_scene_callback(self, msg):
        self.latest_scene = msg

    def spawn_dice_with_mesh_once(self):
        self.spawn_dice_with_mesh()
        self.spawn_timer.cancel()
        self.get_logger().info("Dice mesh spawned with size: {}".format(self.dice_size))

    def publish_dynamic_transforms(self):
        transforms = []

        for face_id, normal in self.face_normals.items():
            tf = TransformStamped()
            tf.header.stamp = self.get_clock().now().to_msg()
            tf.header.frame_id = "dice_rotated_tf"
            tf.child_frame_id = f"face{face_id}_tf"
            offset = (self.dice_size / 2.0) * normal
            tf.transform.translation.x = float(offset[0])
            tf.transform.translation.y = float(offset[1])
            tf.transform.translation.z = float(offset[2])
            q = self.get_quaternion_from_normal(normal)
            tf.transform.rotation.x = q[0]
            tf.transform.rotation.y = q[1]
            tf.transform.rotation.z = q[2]
            tf.transform.rotation.w = q[3]
            transforms.append(tf)

        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = f"face{self.face}_tf"
        tf.child_frame_id = "dice_tf"
        tf.transform.translation.x = 0.0
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = 1.0
        transforms.append(tf)

        self.tf_broadcaster.sendTransform(transforms)

    def update_dice_tf_from_scene(self):
        if not self.dice_spawned or self.latest_scene is None:
            return

        try:
            scene = self.latest_scene

            for obj in scene.world.collision_objects:
                if obj.id == self.dice_name:
                    self.publish_updated_dice_rotated_tf(obj.pose, obj.header.frame_id)
                    return

            for attached_obj in scene.robot_state.attached_collision_objects:
                if attached_obj.object.id == self.dice_name:
                    self.publish_updated_dice_rotated_tf(attached_obj.object.pose, attached_obj.object.header.frame_id)
                    return

            self.get_logger().warn(f"Dice object '{self.dice_name}' not found in scene.")

        except Exception as e:
            self.get_logger().error(f'Error in update_dice_tf_from_scene: {str(e)}')

    def publish_updated_dice_rotated_tf(self, pose: Pose, parent_frame: str):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = parent_frame
        tf.child_frame_id = "dice_rotated_tf"
        tf.transform.translation = Vector3(
            x=pose.position.x,
            y=pose.position.y,
            z=pose.position.z
        )
        tf.transform.rotation = pose.orientation
        self.tf_broadcaster.sendTransform([tf])

    def spawn_dice_with_mesh(self):
        timeout_sec = 10.0
        poll_interval = 0.01
        elapsed = 0.0

        while not self.tf_buffer.can_transform('world', 'dice_rotated_tf', rclpy.time.Time()):
            if elapsed >= timeout_sec:
                self.get_logger().error("Timeout waiting for 'dice_rotated_tf' to become available.")
                return
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=poll_interval))
            elapsed += poll_interval

        pose = PoseStamped()
        pose.header.frame_id = "dice_rotated_tf"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [self.dice_size] * 3

        obj = CollisionObject()
        obj.id = self.dice_name
        obj.header = pose.header
        obj.pose = pose.pose
        obj.primitives = [box]
        obj.primitive_poses = [pose.pose]
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.world.collision_objects = [obj]
        scene.is_diff = True

        req = ApplyPlanningScene.Request(scene=scene)
        future = self.scene_client.call_async(req)
        future.add_done_callback(self.spawn_dice_result)

    def spawn_dice_result(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"ApplyPlanningScene response: {response}")
            self.dice_spawned = response.success
        except Exception as e:
            self.get_logger().error(f'An error occurred while spawning: {str(e)}')

    def get_dice_state_callback(self, request, response):
        try:
            now = rclpy.time.Time()
            z_world = np.array([0, 0, 1])
            best_face = None
            best_dot = -1.0
            best_tf = None
            for face_id in range(1, 7):
                tf = self.tf_buffer.lookup_transform('world', f'face{face_id}_tf', now)
                q = tf.transform.rotation
                q_np = np.array([q.x, q.y, q.z, q.w])
                z_local = np.array([0, 0, 1])
                z_world_face = self.rotate_vector(z_local, q_np)
                dot = np.dot(z_world_face, z_world)
                if dot > best_dot:
                    best_dot = dot
                    best_face = face_id
                    best_tf = tf

            self.face = best_face
            self.dice_face_publisher_.publish(Int16(data=best_face))

            pose = PoseStamped()
            pose.header = best_tf.header
            pose.pose.position = Point(
                x=best_tf.transform.translation.x,
                y=best_tf.transform.translation.y,
                z=best_tf.transform.translation.z
            )
            pose.pose.orientation = best_tf.transform.rotation

            response.pose = pose
            response.face_number = best_face
            response.success = True
            return response

        except Exception as e:
            self.get_logger().error(f"get_dice_state_callback error: {e}")
            response.success = False
            return response

    def get_orientation_for_face(self, face):
        face_to_rpy = {
            1: ( math.pi / 2, 0, 0),
            2: (0, -math.pi / 2, 0),
            3: (0, math.pi, 0),
            4: (0, 0, 0),
            5: (0, math.pi / 2, 0),
            6: (-math.pi / 2, 0, 0),
        }
        rpy = face_to_rpy.get(face, (0, 0, 0))
        return quaternion_from_euler(*rpy)

    def get_quaternion_from_normal(self, normal):
        z_axis = np.array([0, 0, 1])
        v = np.cross(z_axis, normal)
        c = np.dot(z_axis, normal)
        if np.linalg.norm(v) < 1e-6:
            return (0.0, 0.0, 0.0, 1.0) if c > 0 else quaternion_from_euler(math.pi, 0, 0)
        s = math.sqrt((1 + c) * 2)
        vx, vy, vz = v / np.linalg.norm(v)
        return (
            vx * math.sin(math.acos(c) / 2),
            vy * math.sin(math.acos(c) / 2),
            vz * math.sin(math.acos(c) / 2),
            math.cos(math.acos(c) / 2),
        )

    def rotate_vector(self, v, q):
        v_q = (v[0], v[1], v[2], 0.0)
        q_conj = (-q[0], -q[1], -q[2], q[3])
        result = quaternion_multiply(quaternion_multiply(q, v_q), q_conj)
        return result[:3]


def main(args=None):
    rclpy.init(args=args)
    node = DiceSpawner()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()
