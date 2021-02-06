import random
import cv2 
import gym 
import gym_super_mario_bros 
from gym_super_mario_bros import RIGHT_ONLY

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Returns only every 'skip' -th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self.__obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

print(env.observation_space.shape) # Dimensions of a frame
print(env.action_space.n) # Number of actions agent can take

def make_env(env):
    """ Just modifies some gym settings for the game """
    env = MaxAndSkipEnv(env) # Every action the a agent makes is repeated over 4 frames
    env = ProcessFrame84(env) # The size of each frame is reduced to 84x84
    env = ImageToPytorch(env) # The frames are converted to Pytorch Tensors
    env = BufferWrapper(env, 4) # Only every fourth frame is collected by the buffer
    env = ScaledFloatFrame(env) # The frames are normalized so that pixel values are between 0 and 1
    return JoypadSpace(env, RIGHT_ONLY) # The number of actions is redued to 5 (such that the agent can only
                                        # move right)

print(make_env(env))
