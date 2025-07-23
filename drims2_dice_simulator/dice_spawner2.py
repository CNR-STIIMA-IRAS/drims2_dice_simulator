import os
import math
import random
import numpy as np
import trimesh

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
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
        else:
            self.position = Point(x=default_position[0], y=default_position[1], z=default_position[2])

        package_path = get_package_share_directory('drims2_dice_simulator')
        self.dice_mesh_path = os.path.join(package_path, 'urdf', 'Die-OBJ.obj')

        self.tf_broadcaster = TransformBroadcaster(self)
        self.scene_client = self.create_client(ApplyPlanningScene, '/apply_planning_scene')
        while not self.scene_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /apply_planning_scene service...')

        self.dice_face_publisher_ = self.create_publisher(Int16, '/dice_face', 10)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.srv = self.create_service(DiceIdentification, '/dice_identification', self.get_dice_state_callback)

        self.face_normals = {
            1: np.array([ 0,  1,  0]),
            2: np.array([ 1,  0,  0]),
            3: np.array([ 0,  0, -1]),
            4: np.array([ 0,  0,  1]),
            5: np.array([-1,  0,  0]),
            6: np.array([ 0, -1,  0]),
        }

        # Start dynamic TF publishing before spawning the mesh
        self.timer = self.create_timer(0.01, self.publish_dynamic_transforms)
        rclpy.spin_once(self, timeout_sec=0.5)

        # self.spawn_dice_with_mesh()
        self.dice_face_publisher_.publish(Int16(data=self.face))

    def get_dice_state_callback(self, request, response):
        try:
            now = rclpy.time.Time()
            z_world = np.array([0, 0, 1])
            best_dot = -1.0
            best_face = None
            best_tf = None

            # Find which face has its local Z most aligned with world Z+
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
                    best_tf = tf

            # Update internal state and publish face
            self.face = best_face
            self.dice_face_publisher_.publish(Int16(data=self.face))

            # Set response with pose of faceX_tf (in world)
            pose_msg = PoseStamped()
            pose_msg.header = best_tf.header
            pose_msg.pose.position = best_tf.transform.translation
            pose_msg.pose.orientation = best_tf.transform.rotation

            response.face_number = best_face
            response.pose = pose_msg
            response.success = True
            return response

        except Exception as e:
            self.get_logger().error(f"Error in get_dice_state_callback: {str(e)}")
            response.success = False
            return response

    def compute_transformations(self):
        transforms = {}
        now = self.get_clock().now().to_msg()

        tf_base = TransformStamped()
        tf_base.header.stamp = now
        tf_base.header.frame_id = "world"
        tf_base.child_frame_id = "dice_base_tf"
        tf_base.transform.translation.x = self.position.x
        tf_base.transform.translation.y = self.position.y
        tf_base.transform.translation.z = self.position.z
        tf_base.transform.rotation.w = 1.0
        transforms["world->dice_base_tf"] = tf_base

        q_rot = self.get_orientation_for_face(self.face)
        tf_rot = TransformStamped()
        tf_rot.header.stamp = now
        tf_rot.header.frame_id = "dice_base_tf"
        tf_rot.child_frame_id = "dice_rotated_tf"
        tf_rot.transform.rotation.x = q_rot[0]
        tf_rot.transform.rotation.y = q_rot[1]
        tf_rot.transform.rotation.z = q_rot[2]
        tf_rot.transform.rotation.w = q_rot[3]
        transforms["dice_base_tf->dice_rotated_tf"] = tf_rot

        for face_id, normal in self.face_normals.items():
            offset = (self.dice_size / 2.0) * normal
            q = self.get_quaternion_from_normal(normal)
            tf = TransformStamped()
            tf.header.stamp = now
            tf.header.frame_id = "dice_rotated_tf"
            tf.child_frame_id = f"face{face_id}_tf"
            tf.transform.translation.x = offset[0]
            tf.transform.translation.y = offset[1]
            tf.transform.translation.z = offset[2]
            tf.transform.rotation.x = q[0]
            tf.transform.rotation.y = q[1]
            tf.transform.rotation.z = q[2]
            tf.transform.rotation.w = q[3]
            transforms[f"dice_rotated_tf->face{face_id}_tf"] = tf

        tf_alias = TransformStamped()
        tf_alias.header.stamp = now
        tf_alias.header.frame_id = f"face{self.face}_tf"
        tf_alias.child_frame_id = "dice_tf"
        tf_alias.transform.rotation.w = 1.0
        transforms[f"face{self.face}_tf->dice_tf"] = tf_alias

        return transforms

    def invert_transform(self, tf):
        inv = TransformStamped()
        inv.header.stamp = tf.header.stamp
        inv.header.frame_id = tf.child_frame_id
        inv.child_frame_id = tf.header.frame_id

        t = tf.transform.translation
        q = tf.transform.rotation

        q_inv = [-q.x, -q.y, -q.z, q.w]
        trans = np.array([t.x, t.y, t.z])
        q_array = np.array([q.x, q.y, q.z, q.w])
        inv_trans = -self.rotate_vector(trans, q_array)

        inv.transform.translation.x = inv_trans[0]
        inv.transform.translation.y = inv_trans[1]
        inv.transform.translation.z = inv_trans[2]
        inv.transform.rotation.x = q_inv[0]
        inv.transform.rotation.y = q_inv[1]
        inv.transform.rotation.z = q_inv[2]
        inv.transform.rotation.w = q_inv[3]

        return inv

    def publish_dynamic_transforms(self):
        transforms_dict = self.compute_transformations()
        now = self.get_clock().now().to_msg()

        # # Re-evaluate every cycle whether someone else is publishing dice_tf
        # try:
        #     tf = self.tf_buffer.lookup_transform("world", "dice_tf", rclpy.time.Time(), timeout=rclpy.duration.Duration(seconds=0.01))
        #     # If the transform is not recent enough, consider it stale
        #     age = self.get_clock().now() - rclpy.time.Time.from_msg(tf.header.stamp)
        #     if age.nanoseconds > 1e8:  # 100ms timeout
        #         self.external_tf_detected = False
        #     else:
        #         self.external_tf_detected = True
        # except:
        #     # Transform not found → assume no one is publishing
        #     self.external_tf_detected = False
        
        self.external_tf_detected = False

        all_transforms = []

        # If no external dice_tf, compute and publish world → dice_tf
        if not self.external_tf_detected:
            tf_chain = [
                transforms_dict["world->dice_base_tf"],
                transforms_dict["dice_base_tf->dice_rotated_tf"],
                transforms_dict[f"dice_rotated_tf->face{self.face}_tf"],
                transforms_dict[f"face{self.face}_tf->dice_tf"]
            ]

            t_total = np.array([0.0, 0.0, 0.0])
            q_total = np.array([0.0, 0.0, 0.0, 1.0])

            for tf in tf_chain:
                t = tf.transform.translation
                q = np.array([tf.transform.rotation.x, tf.transform.rotation.y,
                            tf.transform.rotation.z, tf.transform.rotation.w])
                trans = np.array([t.x, t.y, t.z])
                t_total += self.rotate_vector(trans, q_total)
                q_total = quaternion_multiply(q_total, q)

            tf_dice_tf = TransformStamped()
            tf_dice_tf.header.stamp = now
            tf_dice_tf.header.frame_id = "world"
            tf_dice_tf.child_frame_id = "dice_tf"
            tf_dice_tf.transform.translation.x = t_total[0]
            tf_dice_tf.transform.translation.y = t_total[1]
            tf_dice_tf.transform.translation.z = t_total[2]
            tf_dice_tf.transform.rotation.x = q_total[0]
            tf_dice_tf.transform.rotation.y = q_total[1]
            tf_dice_tf.transform.rotation.z = q_total[2]
            tf_dice_tf.transform.rotation.w = q_total[3]
            all_transforms.append(tf_dice_tf)

        # Always publish all transforms (except world->*) inverted, preserving parent-child relations
        for key, tf in transforms_dict.items():
            parent = tf.header.frame_id
            child = tf.child_frame_id
            if parent == "world":
                # print(f"Skipping world transform: {key}")
                continue 
            inv = self.invert_transform(tf)
            inv.header.stamp = now
            inv.header.frame_id = child   # new parent becomes child
            inv.child_frame_id = parent   # new child becomes parent
            print(f"Publishing inverted transform: {inv.header.frame_id} -> {inv.child_frame_id}")
            all_transforms.append(inv)

        self.tf_broadcaster.sendTransform(all_transforms)

    def spawn_dice_with_mesh(self):
        pose = PoseStamped()
        pose.header.frame_id = "dice_rotated_tf"
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
            self.get_logger().info(f"Spawned dice with face {self.face} at position {self.position}")
        else:
            self.get_logger().error("Failed to spawn dice in planning scene.")

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
        q_conj = np.array([-q[0], -q[1], -q[2], q[3]])
        v_quat = np.array([v[0], v[1], v[2], 0.0])
        result = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)[:3]
        return np.array(result)

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


def main(args=None):
    rclpy.init(args=args)
    node = DiceSpawner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
