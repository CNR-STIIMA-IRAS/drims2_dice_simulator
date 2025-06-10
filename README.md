# drims2_dice_simulator
The DRIMS2 Summer School Dice Simulator package enables spawning a dice in a robotic cell, with the option to display either a random or a specified face up. Additionally, it publishes a tf at the center of the face-up side.

To launch the simulator, run the following command:

```bash
ros2 launch drims2_dice_simulator spawn_dice.launch.py
```

You can customize the behavior of the dice simulator by passing the following parameters to the launch file:

- `face_up`: Specifies the face of the dice that should be facing up (a value between 1 and 6). If set to 0, or if not provided, the face-up will be chosen randomly.
- `dice_size`: The length of the dice edges in centimeters. The default size is 5 cm.
- `position`: position of the dice with respect to `world`, default [0,0,0.85].