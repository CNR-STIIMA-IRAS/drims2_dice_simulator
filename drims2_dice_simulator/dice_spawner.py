import math
import os
import random
import numpy as np
import trimesh

from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.duration import Duration
from geometry_msgs.msg import PoseStamped, TransformStamped, Point, Vector3
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster, Buffer, TransformListener
from tf_transformations import quaternion_from_euler, quaternion_multiply
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject, PlanningSceneComponents
from shape_msgs.msg import Mesh, MeshTriangle
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

        self.get_scene_client = self.create_client(GetPlanningScene, '/get_planning_scene')
        while not self.get_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /get_planning_scene service...')

        self.face_normals = {
            1: np.array([0, 1, 0]),
            2: np.array([1, 0, 0]),
            3: np.array([0, 0, -1]),
            4: np.array([0, 0, 1]),
            5: np.array([-1, 0, 0]),
            6: np.array([0, -1, 0]),
        }

        self.publish_base_tf()
        self.dynamic_tf_timer = self.create_timer(0.05, self.publish_dynamic_transforms)
        self.scene_tracking_timer = self.create_timer(0.1, self.update_dice_tf_from_scene)
        # self.spawn_dice_with_mesh()
        self.spawn_timer = self.create_timer(0.5, self.spawn_dice_with_mesh_once)

        self.dice_face_publisher_.publish(Int16(data=self.face))
        
    def spawn_dice_with_mesh_once(self):
        self.spawn_dice_with_mesh()
        self.spawn_timer.cancel()  # one-shot: stop timer
        self.get_logger().info("Dice mesh spawned with size: {}".format(self.dice_size))

    def publish_base_tf(self):
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = "world"
        tf.child_frame_id = "dice_base_tf"
        tf.transform.translation = Vector3(x=self.position.x, y=self.position.y, z=self.position.z)
        tf.transform.rotation.x = 0.0
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        tf.transform.rotation.w = 1.0
        self.static_tf_broadcaster.sendTransform([tf])

    def publish_dynamic_transforms(self):
        transforms = []

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
        transforms.append(tf1)

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
        self.get_logger().info(f"Published transforms for dice face {self.face}")

    def dice_spawned(self, future):
        try:
            response = future.result()
            if response.error_code.val == response.error_code.SUCCESS:
                self.get_logger().info(f"Dice spawned successfully with face {self.face}")
            else:
                self.get_logger().error(f"Failed to spawn dice: {response.error_code.val}")
        except Exception as e:
            self.get_logger().error(f'An error occurred while spawning: {str(e)}')

    def send_tf(self, future):
        try:
            response = future.result()
            scene = response.scene
            for obj in scene.world.collision_objects:
                if obj.id == self.dice_name and obj.mesh_poses:
                    pose = obj.mesh_poses[0]
                    tf = TransformStamped()
                    tf.header.stamp = self.get_clock().now().to_msg()
                    tf.header.frame_id = "dice_base_tf"
                    tf.child_frame_id = "dice_rotated_tf"
                    tf.transform.translation = pose.position
                    tf.transform.rotation = pose.orientation
                    self.tf_broadcaster.sendTransform([tf])
                    break
        except Exception as e:
            self.get_logger().error(f'An error occurred while deleting: {str(e)}')

        self.get_logger().info(f"update_dice_tf_from_scene")

    def update_dice_tf_from_scene(self):
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES

        future = self.get_scene_client.call_async(req)
        future.add_done_callback(self.send_tf)
        # rclpy.spin_until_future_complete(self, future, timeout_sec=1.0)

        # if not future.result():
        #     self.get_logger().warn("No result from /get_planning_scene")
        #     return

        # scene = future.result().scene
        # for obj in scene.world.collision_objects:
        #     if obj.id == self.dice_name and obj.mesh_poses:
        #         pose = obj.mesh_poses[0]
        #         tf = TransformStamped()
        #         tf.header.stamp = self.get_clock().now().to_msg()
        #         tf.header.frame_id = "dice_base_tf"
        #         tf.child_frame_id = "dice_rotated_tf"
        #         tf.transform.translation = pose.position
        #         tf.transform.rotation = pose.orientation
        #         self.tf_broadcaster.sendTransform([tf])
        #         break

        self.get_logger().info(f"update_dice_tf_from_scene")


    def spawn_dice_with_mesh(self):
        # Wait for 'dice_rotated_tf' to become available in the TF tree
        timeout_sec = 5.0
        poll_interval = 0.01
        elapsed = 0.0

        while not self.tf_buffer.can_transform('world', 'dice_rotated_tf', rclpy.time.Time()):
            if elapsed >= timeout_sec:
                self.get_logger().error("Timeout waiting for 'dice_rotated_tf' to become available.")
                return
            self.get_logger().info("Waiting for 'dice_rotated_tf' to become available...")
            self.get_clock().sleep_for(rclpy.duration.Duration(seconds=poll_interval))
            elapsed += poll_interval

        # Prepare pose relative to 'dice_rotated_tf'
        pose = PoseStamped()
        pose.header.frame_id = "dice_rotated_tf"
        pose.pose.orientation.w = 1.0

        # Load and scale mesh
        mesh = trimesh.load(self.dice_mesh_path, force='mesh')
        mesh_msg = Mesh()
        for tri in mesh.faces:
            mesh_msg.triangles.append(MeshTriangle(vertex_indices=tri.tolist()))
        for v in mesh.vertices:
            point = Point()
            point.x, point.y, point.z = v * self.dice_size
            mesh_msg.vertices.append(point)

        # Create collision object
        obj = CollisionObject()
        obj.id = self.dice_name
        obj.header = pose.header
        obj.meshes = [mesh_msg]
        obj.mesh_poses = [pose.pose]
        obj.operation = CollisionObject.ADD

        # Apply to planning scene
        scene = PlanningScene()
        scene.world.collision_objects = [obj]
        scene.is_diff = True

        req = ApplyPlanningScene.Request(scene=scene)
        future = self.scene_client.call_async(req)
        future.add_done_callback(self.dice_spawned)
        # rclpy.spin_until_future_complete(self, future)

    def get_dice_state_callback(self, request, response):
        try:
            now = rclpy.time.Time()
            z_world = np.array([0, 0, 1])
            best_face = None
            best_dot = -1.0
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

            self.face = best_face
            self.dice_face_publisher_.publish(Int16(data=best_face))

            tf = self.tf_buffer.lookup_transform('world', 'dice_tf', now)
            pose = PoseStamped()
            pose.header = tf.header
            pose.pose.position = tf.transform.translation
            pose.pose.orientation = tf.transform.rotation
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
