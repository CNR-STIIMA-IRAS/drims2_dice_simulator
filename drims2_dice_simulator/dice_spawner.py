import math
import os
import time
import random
import trimesh
import numpy as np

from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import Pose, PoseStamped, TransformStamped, Point, Vector3
from tf2_ros import StaticTransformBroadcaster, Buffer, TransformListener
from tf_transformations import quaternion_from_euler, quaternion_multiply
from moveit_msgs.srv import ApplyPlanningScene, GetPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle
from std_msgs.msg import Int16

from drims2_msgs.srv import DiceIdentification, AttachObject


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
        self.dice_mesh_path = os.path.join(package_path, 'urdf', 'simplify_Die-OBJ.obj')

        self.internal_node = Node('dice_spawner_internal_node')
        self.internal_executor = MultiThreadedExecutor(num_threads=4)
        self.internal_executor.add_node(self.internal_node)

        self.service_callback_group = ReentrantCallbackGroup()
        self.get_scene_callback_group = ReentrantCallbackGroup()

        self.srv = self.create_service(
            DiceIdentification,
            '/dice_identification',
            self.get_dice_state_callback,
            callback_group=self.service_callback_group
        )

        self.dice_face_publisher_ = self.create_publisher(Int16, '/dice_face', 10)

        self.apply_scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.apply_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /apply_planning_scene service...')

        self.get_scene_client = self.internal_node.create_client(
            GetPlanningScene,
            '/get_planning_scene',
            callback_group=self.get_scene_callback_group
        )
        while not self.get_scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /get_planning_scene service...")

        self.add_client = self.create_client(AttachObject, '/attach_object')
        while not self.add_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for /attach_object service...")

        self.face_normals = {
            1: np.array([0, 1, 0]),
            2: np.array([1, 0, 0]),
            3: np.array([0, 0, -1]),
            4: np.array([0, 0, 1]),
            5: np.array([-1, 0, 0]),
            6: np.array([0, -1, 0]),
        }

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        try:
            # Dummy lookup to check if "world" exists (even self -> self transform)
            self.tf_buffer.lookup_transform("world", "world", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=1.0))
            self.get_logger().info("TF frame 'world' found.")
            self.world = "world"
        except Exception:
            self.get_logger().info("TF frame 'world' not found. Falling back to 'base_footprint'.")
            self.world = "base_footprint"

        self.publish_all_static_transforms()
        time.sleep(1.0)
        self.spawn_dice_with_mesh()
        self.dice_face_publisher_.publish(Int16(data=self.face))

    def publish_all_static_transforms(self):
        transforms = []

        tf_base = TransformStamped()
        tf_base.header.stamp = self.get_clock().now().to_msg()
        tf_base.header.frame_id = self.world
        tf_base.child_frame_id = "dice_base_tf"
        tf_base.transform.translation = Vector3(
            x=self.position.x,
            y=self.position.y,
            z=self.position.z
        )
        tf_base.transform.rotation.x = 0.0
        tf_base.transform.rotation.y = 0.0
        tf_base.transform.rotation.z = 0.0
        tf_base.transform.rotation.w = 1.0
        transforms.append(tf_base)

        q = self.get_orientation_for_face(self.face)
        tf_rot = TransformStamped()
        tf_rot.header.stamp = self.get_clock().now().to_msg()
        tf_rot.header.frame_id = "dice_base_tf"
        tf_rot.child_frame_id = "dice_rotated_tf"
        tf_rot.transform.translation.x = 0.0
        tf_rot.transform.translation.y = 0.0
        tf_rot.transform.translation.z = 0.0
        tf_rot.transform.rotation.x = q[0]
        tf_rot.transform.rotation.y = q[1]
        tf_rot.transform.rotation.z = q[2]
        tf_rot.transform.rotation.w = q[3]
        transforms.append(tf_rot)

        for face_id, normal in self.face_normals.items():
            offset = (self.dice_size / 2.0) * normal
            q_face = self.get_quaternion_from_normal(normal)
            tf_face = TransformStamped()
            tf_face.header.stamp = self.get_clock().now().to_msg()
            tf_face.header.frame_id = "dice_rotated_tf"
            tf_face.child_frame_id = f"face{face_id}_tf"
            tf_face.transform.translation.x = float(offset[0])
            tf_face.transform.translation.y = float(offset[1])
            tf_face.transform.translation.z = float(offset[2])
            tf_face.transform.rotation.x = q_face[0]
            tf_face.transform.rotation.y = q_face[1]
            tf_face.transform.rotation.z = q_face[2]
            tf_face.transform.rotation.w = q_face[3]
            transforms.append(tf_face)

        tf_dice = TransformStamped()
        tf_dice.header.stamp = self.get_clock().now().to_msg()
        tf_dice.header.frame_id = f"face{self.face}_tf"
        tf_dice.child_frame_id = "dice_tf"
        tf_dice.transform.translation.x = 0.0
        tf_dice.transform.translation.y = 0.0
        tf_dice.transform.translation.z = 0.0
        tf_dice.transform.rotation.x = 0.0
        tf_dice.transform.rotation.y = 0.0
        tf_dice.transform.rotation.z = 0.0
        tf_dice.transform.rotation.w = 1.0
        transforms.append(tf_dice)

        self.static_tf_broadcaster.sendTransform(transforms)

    def update_dice_tf_from_scene(self):
        try:
            request = GetPlanningScene.Request()
            request.components.components = (
                GetPlanningScene.Request().components.SCENE_SETTINGS |
                GetPlanningScene.Request().components.WORLD_OBJECT_NAMES |
                GetPlanningScene.Request().components.WORLD_OBJECT_GEOMETRY |
                GetPlanningScene.Request().components.ROBOT_STATE_ATTACHED_OBJECTS
            )

            future = self.get_scene_client.call_async(request)
            self.internal_executor.spin_until_future_complete(future, timeout_sec=5.0)

            if not future.done():
                self.get_logger().warn("Timeout while waiting for planning scene.")
                return False

            result = future.result()

            for obj in result.scene.world.collision_objects:
                if obj.id == self.dice_name:
                    self.publish_updated_dice_rotated_tf(obj.pose, obj.header.frame_id)
                    return True

            for attached_obj in result.scene.robot_state.attached_collision_objects:
                if attached_obj.object.id == self.dice_name:
                    self.publish_updated_dice_rotated_tf(attached_obj.object.pose, attached_obj.object.header.frame_id)
                    return True

            self.get_logger().warn(f"Dice object '{self.dice_name}' not found in planning scene.")
            return False

        except Exception as e:
            self.get_logger().error(f'Error in update_dice_tf_from_scene: {str(e)}')
            return False

    def publish_updated_dice_rotated_tf(self, pose: Pose, parent_frame: str):
        transforms = []

        tf_rot = TransformStamped()
        tf_rot.header.stamp = self.get_clock().now().to_msg()
        tf_rot.header.frame_id = parent_frame
        tf_rot.child_frame_id = "dice_rotated_tf"
        tf_rot.transform.translation = Vector3(
            x=pose.position.x,
            y=pose.position.y,
            z=pose.position.z
        )
        tf_rot.transform.rotation = pose.orientation
        transforms.append(tf_rot)

        for face_id, normal in self.face_normals.items():
            offset = (self.dice_size / 2.0) * normal
            q = self.get_quaternion_from_normal(normal)
            tf_face = TransformStamped()
            tf_face.header.stamp = self.get_clock().now().to_msg()
            tf_face.header.frame_id = "dice_rotated_tf"
            tf_face.child_frame_id = f"face{face_id}_tf"
            tf_face.transform.translation.x = float(offset[0])
            tf_face.transform.translation.y = float(offset[1])
            tf_face.transform.translation.z = float(offset[2])
            tf_face.transform.rotation.x = q[0]
            tf_face.transform.rotation.y = q[1]
            tf_face.transform.rotation.z = q[2]
            tf_face.transform.rotation.w = q[3]
            transforms.append(tf_face)

        self.static_tf_broadcaster.sendTransform(transforms)

    def spawn_dice_with_mesh(self):
        pose = PoseStamped()
        pose.header.frame_id = "dice_rotated_tf"
        pose.pose.position.y -= 0.004
        pose.pose.orientation.w = 1.0

        mesh = trimesh.load(self.dice_mesh_path, force='mesh')
        mesh_msg = Mesh()
        for tri in mesh.faces:
            mesh_msg.triangles.append(MeshTriangle(vertex_indices=tri.tolist()))
        for v in mesh.vertices:
            point = Point()
            point.x, point.y, point.z = v * self.dice_size
            mesh_msg.vertices.append(point)

        obj = CollisionObject()
        obj.id = self.dice_name
        obj.header = pose.header
        obj.meshes = [mesh_msg]
        obj.mesh_poses = [pose.pose]
        obj.operation = CollisionObject.ADD

        scene = PlanningScene()
        scene.world.collision_objects = [obj]
        scene.is_diff = True

        req = ApplyPlanningScene.Request(scene=scene)
        future = self.apply_scene_client.call_async(req)
        future.add_done_callback(self.spawn_dice_result)

        self.get_logger().info(f"Spawned dice with:\n - face {self.face} up \n - position [{self.position.x}, {self.position.y}, {self.position.z}] \n - size {self.dice_size}")

    def spawn_dice_result(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"AddObject response: {response}")
        except Exception as e:
            self.get_logger().error(f'Error while spawning dice: {str(e)}')

    def get_dice_state_callback(self, request, response):
        self.get_logger().info("Received dice identification request")
        try:
            success = self.update_dice_tf_from_scene()
            if not success:
                self.get_logger().warn("Failed to update dice transform from planning scene.")
                response.success = False
                return response

            time.sleep(0.5)

            now = rclpy.time.Time()
            z_world = np.array([0, 0, 1])
            best_face = None
            best_dot = -1.0
            best_tf = None

            for face_id in range(1, 7):
                tf = self.tf_buffer.lookup_transform(self.world, f'face{face_id}_tf', now)
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

            self.get_logger().info(f"Detected face up: {best_face}")
            self.get_logger().info(f"Position: x={pose.pose.position.x:.3f}, y={pose.pose.position.y:.3f}, z={pose.pose.position.z:.3f}")
            self.get_logger().info(f"Orientation (quaternion): x={pose.pose.orientation.x:.3f}, y={pose.pose.orientation.y:.3f}, z={pose.pose.orientation.z:.3f}, w={pose.pose.orientation.w:.3f}")

            dice_tf = TransformStamped()
            dice_tf.header.stamp = self.get_clock().now().to_msg()
            dice_tf.header.frame_id = f"face{self.face}_tf"
            dice_tf.child_frame_id = "dice_tf"
            dice_tf.transform.translation.x = 0.0
            dice_tf.transform.translation.y = 0.0
            dice_tf.transform.translation.z = 0.0
            dice_tf.transform.rotation.x = 0.0
            dice_tf.transform.rotation.y = 0.0
            dice_tf.transform.rotation.z = 0.0
            dice_tf.transform.rotation.w = 1.0
            self.static_tf_broadcaster.sendTransform([dice_tf])

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
            1: (math.pi / 2, 0, 0),
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
