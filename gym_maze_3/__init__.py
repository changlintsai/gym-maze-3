from gym.envs.registration import register

register(
    id='maze-v3',
    entry_point='gym_maze_3.envs:MazeEnv3',
)
