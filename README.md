# drims2_dice_simulator

A lightweight ROS 2 package for spawning a dice inside a robotic cell. You can choose a specific face to be up or let it be selected at random. The node also publishes a `tf` frame centered on the currently face-up side.

## Launch

```bash
ros2 launch drims2_dice_simulator spawn_dice.launch.py
```

## Parameters

- **`face_up`** *(int, 0–6, default: `0`)*  
  Face that should point upward. Use `1`–`6` to set the face up; `0` (or unset) picks a random face.

- **`dice_size`** *(float, cm, default: `5.0`)*  
  Edge length of the dice.

- **`position`** *(float[3], default: `[-0.1, 0.0, 0.85]`)*  
  XYZ spawn position in the `world` frame.

## Examples

Random face, default size and position:
```bash
ros2 launch drims2_dice_simulator spawn_dice.launch.py
```

Fixed face (e.g., “6”) and a smaller dice:
```bash
ros2 launch drims2_dice_simulator spawn_dice.launch.py face_up:=6 dice_size:=2.5
```

Spawn position tuned for PAL TIAGo Pro:
```bash
ros2 launch drims2_dice_simulator spawn_dice.launch.py position:=[0.7, 0.0, 0.85]
```

> The provided defaults work well for UR10e and ABB YuMi workcells.

**TF output:** a frame is published at the center of the upward face for easy grasp planning.
