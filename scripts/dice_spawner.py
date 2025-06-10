import rclpy
import random
import math
import os
import trimesh
import numpy as np

from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from tf2_ros import StaticTransformBroadcaster
from tf_transformations import quaternion_from_euler, quaternion_multiply

from moveit_msgs.srv import ApplyPlanningScene
from moveit_msgs.msg import PlanningScene, CollisionObject
from shape_msgs.msg import Mesh, MeshTriangle

from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Int16

class DiceSpawner(Node):
    def __init__(self):
        super().__init__('dice_spawner_node')

        # Parameters
        self.declare_parameter("face_up", 0)
        face = self.get_parameter("face_up").get_parameter_value().integer_value
        self.face = face if 1 <= face <= 6 else random.randint(1, 6)

        self.dice_name = "dice"
        self.declare_parameter("dice_size", 0.02)
        self.dice_size = self.get_parameter("dice_size").get_parameter_value().double_value

        default_position = [0.5, 0.0, 0.85]
        self.declare_parameter("position", default_position)
        self.position = self.get_parameter("position").get_parameter_value().double_array_value
        if len(self.position) != 3:
            self.get_logger().warn(f"Invalid position parameter, using default {list(default_position)}.")
            self.position = default_position
        else:
            self.get_logger().info(f"Using position {list(self.position)}.")

        # Path to mesh file
        package_path = get_package_share_directory('drims2_dice_simulator')
        self.dice_mesh_path = os.path.join(package_path, 'urdf', 'Die-OBJ.obj')

        # TF broadcaster
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)

        # Planning scene service
        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /apply_planning_scene service...')

        # Dice face publisher
        self.dice_face_publisher_ = self.create_publisher(Int16, '/dice_face', 10)

        # Publish transforms and spawn dice
        self.publish_static_transforms()
        self.spawn_dice_with_mesh()

        dice_face_msg = Int16()
        dice_face_msg.data = self.face
        self.dice_face_publisher_.publish(dice_face_msg)

    def publish_static_transforms(self):
        # --- Base TF (position only) ---
        base_tf = TransformStamped()
        base_tf.header.stamp = self.get_clock().now().to_msg()
        base_tf.header.frame_id = "world"
        base_tf.child_frame_id = "dice_base_tf"
        base_tf.transform.translation.x = self.position[0]
        base_tf.transform.translation.y = self.position[1]
        base_tf.transform.translation.z = self.position[2]
        base_tf.transform.rotation.x = 0.0
        base_tf.transform.rotation.y = 0.0
        base_tf.transform.rotation.z = 0.0
        base_tf.transform.rotation.w = 1.0

        # --- Rotation TF to align desired face up ---
        q = self.get_orientation_for_face(self.face)
        rotated_tf = TransformStamped()
        rotated_tf.header.stamp = self.get_clock().now().to_msg()
        rotated_tf.header.frame_id = "dice_base_tf"
        rotated_tf.child_frame_id = "dice_rotated_tf"
        rotated_tf.transform.translation.x = 0.0
        rotated_tf.transform.translation.y = 0.0
        rotated_tf.transform.translation.z = 0.0
        rotated_tf.transform.rotation.x = q[0]
        rotated_tf.transform.rotation.y = q[1]
        rotated_tf.transform.rotation.z = q[2]
        rotated_tf.transform.rotation.w = q[3]

        # --- Frame on center face up ---
        dice_tf = TransformStamped()
        dice_tf.header.stamp = self.get_clock().now().to_msg()
        dice_tf.header.frame_id = "dice_base_tf"
        dice_tf.child_frame_id = "dice_tf"

        dice_tf.transform.translation.x = 0.0
        dice_tf.transform.translation.y = 0.0
        dice_tf.transform.translation.z = self.dice_size / 2.0 
        dice_tf.transform.rotation.x = 0.0
        dice_tf.transform.rotation.y = 0.0
        dice_tf.transform.rotation.z = 0.0
        dice_tf.transform.rotation.w = 1.0

        # --- Publish all ---
        self.static_tf_broadcaster.sendTransform([
            base_tf,
            rotated_tf,
            dice_tf,
        ])

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
            self.get_logger().info(f"Spawned dice with face {self.face} up at position {list(self.position)}")
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
        # Adjusted RPY values to make each face face upward (+Z)
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

