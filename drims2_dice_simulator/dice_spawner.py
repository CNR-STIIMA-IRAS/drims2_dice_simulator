import os
import math
import random
import numpy as np
import trimesh

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from tf2_ros import StaticTransformBroadcaster, Buffer, TransformListener
from tf_transformations import quaternion_from_euler, quaternion_multiply

from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle

from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Int16
from drims2_msgs.srv import DiceIdentification

class DiceSpawner(Node):
    def __init__(self):
        super().__init__('dice_spawner_node')

        # Read parameters
        self.declare_parameter("face_up", 0)
        face = self.get_parameter("face_up").get_parameter_value().integer_value
        self.face = face if 1 <= face <= 6 else random.randint(1, 6)

        self.dice_name = "dice"
        self.declare_parameter("dice_size", 0.02)
        self.dice_size = self.get_parameter("dice_size").get_parameter_value().double_value

        default_position = [0.5, 0.0, 0.85]
        self.declare_parameter("position", default_position)
        pos_param = self.get_parameter("position").get_parameter_value().double_array_value

        if len(pos_param) == 3:
            self.position = Point(x=pos_param[0], y=pos_param[1], z=pos_param[2])
            self.get_logger().info(f"Using position {[self.position.x, self.position.y, self.position.z]}")
        else:
            self.get_logger().warn(f"Invalid position parameter, using default {list(default_position)}.")
            self.position = Point(x=default_position[0], y=default_position[1], z=default_position[2])

        # Path to dice mesh
        package_path = get_package_share_directory('drims2_dice_simulator')
        self.dice_mesh_path = os.path.join(package_path, 'urdf', 'Die-OBJ.obj')

        # TF broadcaster
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # MoveIt planning scene service client
        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /apply_planning_scene service...')

        # Publisher for current face up
        self.dice_face_publisher_ = self.create_publisher(Int16, '/dice_face', 10)

        # TF buffer and listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Service to get dice state
        self.srv = self.create_service(DiceIdentification, '/dice_identification', self.get_dice_state_callback)

        # Define local face normals in dice_rotated_tf frame
        self.face_normals = {
            1: np.array([ 0,  1,  0]),
            2: np.array([ 1,  0,  0]),
            3: np.array([ 0,  0, -1]),
            4: np.array([ 0,  0,  1]),
            5: np.array([-1,  0,  0]),
            6: np.array([ 0, -1,  0]),
        }

        # Publish transforms and spawn dice mesh
        self.publish_static_transforms()
        self.spawn_dice_with_mesh()

        # Publish initial face value
        self.dice_face_publisher_.publish(Int16(data=self.face))

    def get_dice_state_callback(self, request, response):
        self.get_logger().info("--------------------------------")
        self.get_logger().info("Received get_dice_state request")

        try:
            now = rclpy.time.Time()
            z_world = np.array([0, 0, 1])
            best_dot = -1.0
            best_face = None

            # Find the face whose Z axis aligns best with world Z
            for face_id in range(1, 7):
                tf = self.tf_buffer.lookup_transform('world', f'face{face_id}_tf', now)
                q = tf.transform.rotation
                q_np = np.array([q.x, q.y, q.z, q.w])
                z_local = np.array([0, 0, 1])
                z_in_world = self.rotate_vector(z_local, q_np)
                dot = np.dot(z_in_world, z_world)
                if dot > best_dot:
                    best_dot = dot
                    best_face = face_id

            # Update dice_tf to alias the current top face
            alias_tf = TransformStamped()
            alias_tf.header.stamp = self.get_clock().now().to_msg()
            alias_tf.header.frame_id = f"face{best_face}_tf"
            alias_tf.child_frame_id = "dice_tf"
            alias_tf.transform.translation.x = 0.0
            alias_tf.transform.translation.y = 0.0
            alias_tf.transform.translation.z = 0.0
            alias_tf.transform.rotation.x = 0.0
            alias_tf.transform.rotation.y = 0.0
            alias_tf.transform.rotation.z = 0.0
            alias_tf.transform.rotation.w = 1.0
            self.static_tf_broadcaster.sendTransform([alias_tf])

            # Get world pose
            tf = self.tf_buffer.lookup_transform('world', 'dice_tf', now)
            pose_msg = PoseStamped()
            pose_msg.header = tf.header
            pose_msg.pose.position = Point(
                x=tf.transform.translation.x,
                y=tf.transform.translation.y,
                z=tf.transform.translation.z
                )
            pose_msg.pose.orientation = tf.transform.rotation
            response.pose = pose_msg
            response.face_number = best_face
            response.success = True

            # Update internal state
            self.face = best_face
            self.dice_face_publisher_.publish(Int16(data=self.face))
            self.position = pose_msg.pose.position

            self.get_logger().info(f"Face up: {response.face_number}")
            position = pose_msg.pose.position
            orientation = pose_msg.pose.orientation

            self.get_logger().info(
                f"Pose:\n"
                f"  Position -> x: {position.x:.6f}, y: {position.y:.6f}, z: {position.z:.6f}\n"
                f"  Orientation (quaternion) -> x: {orientation.x:.6f}, y: {orientation.y:.6f}, "
                f"z: {orientation.z:.6f}, w: {orientation.w:.6f}"
            )

            return response

        except Exception as e:
            self.get_logger().error(f"Error in get_dice_state_callback: {str(e)}")
            response.success = False
            return response

    def publish_static_transforms(self):
        transforms = []

        # world → dice_base_tf
        base_tf = TransformStamped()
        base_tf.header.stamp = self.get_clock().now().to_msg()
        base_tf.header.frame_id = "world"
        base_tf.child_frame_id = "dice_base_tf"
        base_tf.transform.translation.x = self.position.x
        base_tf.transform.translation.y = self.position.y
        base_tf.transform.translation.z = self.position.z
        base_tf.transform.rotation.x = 0.0
        base_tf.transform.rotation.y = 0.0
        base_tf.transform.rotation.z = 0.0
        base_tf.transform.rotation.w = 1.0
        transforms.append(base_tf)

        # dice_base_tf → dice_rotated_tf
        q_rot = self.get_orientation_for_face(self.face)
        rotated_tf = TransformStamped()
        rotated_tf.header.stamp = self.get_clock().now().to_msg()
        rotated_tf.header.frame_id = "dice_base_tf"
        rotated_tf.child_frame_id = "dice_rotated_tf"
        rotated_tf.transform.translation.x = 0.0
        rotated_tf.transform.translation.y = 0.0
        rotated_tf.transform.translation.z = 0.0
        rotated_tf.transform.rotation.x = q_rot[0]
        rotated_tf.transform.rotation.y = q_rot[1]
        rotated_tf.transform.rotation.z = q_rot[2]
        rotated_tf.transform.rotation.w = q_rot[3]
        transforms.append(rotated_tf)

        # dice_rotated_tf → faceX_tf
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

        # faceX_tf → dice_tf
        alias_tf = TransformStamped()
        alias_tf.header.stamp = self.get_clock().now().to_msg()
        alias_tf.header.frame_id = f"face{self.face}_tf"
        alias_tf.child_frame_id = "dice_tf"
        alias_tf.transform.translation.x = 0.0
        alias_tf.transform.translation.y = 0.0
        alias_tf.transform.translation.z = 0.0
        alias_tf.transform.rotation.x = 0.0
        alias_tf.transform.rotation.y = 0.0
        alias_tf.transform.rotation.z = 0.0
        alias_tf.transform.rotation.w = 1.0
        transforms.append(alias_tf)

        self.static_tf_broadcaster.sendTransform(transforms)

    def spawn_dice_with_mesh(self):
        pose = PoseStamped()
        pose.header.frame_id = "dice_rotated_tf"
        pose.pose.position.x = 0.0
        pose.pose.position.y = 0.0
        pose.pose.position.z = 0.0
        pose.pose.orientation.x = 0.0
        pose.pose.orientation.y = 0.0
        pose.pose.orientation.z = 0.0
        pose.pose.orientation.w = 1.0
        self.dice_pose = pose

        mesh_msg = self.load_mesh_to_shape_msg(self.dice_mesh_path)

        collision_obj = CollisionObject()
        collision_obj.id = self.dice_name
        collision_obj.header = pose.header
        collision_obj.meshes = [mesh_msg]
        collision_obj.mesh_poses = [pose.pose]
        collision_obj.operation = CollisionObject.ADD

        planning_scene = PlanningScene()
        planning_scene.world.collision_objects = [collision_obj]
        planning_scene.is_diff = True

        req = ApplyPlanningScene.Request()
        req.scene = planning_scene

        future = self.scene_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() and future.result().success:
            self.get_logger().info(
                f"Spawned dice with face {self.face} up at position [{self.position.x}, {self.position.y}, {self.position.z}]"
            )
        else:
            self.get_logger().error("Failed to spawn dice in planning scene.")

    def load_mesh_to_shape_msg(self, path):
        mesh = trimesh.load(path, force='mesh')
        shape_mesh = Mesh()
        for face in mesh.faces:
            triangle = MeshTriangle()
            triangle.vertex_indices = [int(face[0]), int(face[1]), int(face[2])]
            shape_mesh.triangles.append(triangle)
        for vertex in mesh.vertices:
            point = Point()
            point.x, point.y, point.z = vertex * self.dice_size
            shape_mesh.vertices.append(point)
        return shape_mesh

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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
