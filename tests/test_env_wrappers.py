import numpy as np
import gymnasium as gym

from src.envs import DiscreteCarRacingWrapper, FrameStackLastAxisWrapper


class DummyCarEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(8, 8, 3), dtype=np.uint8)
        self.action_space = gym.spaces.Box(
            low=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.asarray([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.last_action = None
        return np.zeros((8, 8, 3), dtype=np.uint8), {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32)
        obs = np.zeros((8, 8, 3), dtype=np.uint8)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class DummyFrameEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(2, 2, 1), dtype=np.uint8)
        self.action_space = gym.spaces.Discrete(2)
        self._value = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._value = 1
        obs = np.full((2, 2, 1), self._value, dtype=np.uint8)
        return obs, {}

    def step(self, action):
        self._value += 1
        obs = np.full((2, 2, 1), self._value, dtype=np.uint8)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def test_discrete_carracing_wrapper_maps_actions():
    env = DummyCarEnv()
    wrapped = DiscreteCarRacingWrapper(env)
    assert isinstance(wrapped.action_space, gym.spaces.Discrete)
    assert wrapped.action_space.n == wrapped.action_table.shape[0]
    assert wrapped.action_space.n == 7
    assert np.allclose(wrapped.action_table[3], np.asarray([-0.6, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(wrapped.action_table[4], np.asarray([0.6, 0.0, 0.0], dtype=np.float32))
    assert np.allclose(wrapped.action_table[5], np.asarray([-0.6, 0.4, 0.0], dtype=np.float32))
    assert np.allclose(wrapped.action_table[6], np.asarray([0.6, 0.4, 0.0], dtype=np.float32))

    wrapped.reset(seed=0)
    _, _, _, _, _ = wrapped.step(0)
    assert np.allclose(env.last_action, wrapped.action_table[0])

    _, _, _, _, _ = wrapped.step(wrapped.action_space.n - 1)
    assert np.allclose(env.last_action, wrapped.action_table[-1])


def test_discrete_carracing_wrapper_smooths_actions_and_resets():
    env = DummyCarEnv()
    wrapped = DiscreteCarRacingWrapper(env, smooth_beta=0.5)

    wrapped.reset(seed=0)
    _, _, _, _, _ = wrapped.step(1)  # gas
    assert np.allclose(env.last_action, np.asarray([0.0, 0.5, 0.0], dtype=np.float32))

    _, _, _, _, _ = wrapped.step(1)  # gas again
    assert np.allclose(env.last_action, np.asarray([0.0, 0.75, 0.0], dtype=np.float32))

    _, _, _, _, _ = wrapped.step(0)  # coast
    assert np.allclose(env.last_action, np.asarray([0.0, 0.375, 0.0], dtype=np.float32))

    wrapped.reset(seed=1)
    _, _, _, _, _ = wrapped.step(1)  # gas after reset
    assert np.allclose(env.last_action, np.asarray([0.0, 0.5, 0.0], dtype=np.float32))


def test_discrete_carracing_wrapper_invalid_smooth_beta():
    env = DummyCarEnv()
    try:
        _ = DiscreteCarRacingWrapper(env, smooth_beta=-0.1)
        raise AssertionError("Expected ValueError for smooth_beta < 0")
    except ValueError:
        pass
    try:
        _ = DiscreteCarRacingWrapper(env, smooth_beta=1.1)
        raise AssertionError("Expected ValueError for smooth_beta > 1")
    except ValueError:
        pass


def test_frame_stack_wrapper_stacks_last_axis():
    env = DummyFrameEnv()
    wrapped = FrameStackLastAxisWrapper(env, num_stack=4)

    obs0, _ = wrapped.reset(seed=0)
    assert obs0.shape == (2, 2, 4)
    assert np.all(obs0 == 1)

    obs1, _, _, _, _ = wrapped.step(0)
    assert np.all(obs1[..., :3] == 1)
    assert np.all(obs1[..., 3:] == 2)

    obs2, _, _, _, _ = wrapped.step(1)
    assert np.all(obs2[..., :2] == 1)
    assert np.all(obs2[..., 2:3] == 2)
    assert np.all(obs2[..., 3:] == 3)
