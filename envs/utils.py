from gym.envs.registration import load

from .normalized_env import NormalizedActionWrapper


def mujoco_wrapper(entry_point, **kwargs):

    env_cls = load(entry_point)
    env = env_cls(**kwargs)

    env = NormalizedActionWrapper(env)
    return env
