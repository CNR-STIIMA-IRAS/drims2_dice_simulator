from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'face_up',
            default_value='0',
            description='Face number facing upward (1–6, 0 = random)'
        ),
        DeclareLaunchArgument(
            'dice_size',
            default_value='0.027',
            description='Length of the dice edge (in meters)'
        ),
        DeclareLaunchArgument(
            'position',
            default_value='[0.25, 0.0, 0.80]',
            description='Initial dice position [x, y, z]'
        ),

        Node(
            package='drims2_dice_simulator',
            executable='dice_spawner',
            output='screen',
            parameters=[{
                'face_up': LaunchConfiguration('face_up'),
                'dice_size': LaunchConfiguration('dice_size'),
                'position': LaunchConfiguration('position'),
            }]
        )
    ])
