import numpy as np
import gymnasium as gym


class PartialObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, mask_indices: list[int]):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.mask_indices = np.array(mask_indices, dtype=np.int64)
        low = env.observation_space.low.copy()
        high = env.observation_space.high.copy()
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=env.observation_space.dtype)

    def observation(self, observation):
        obs = np.array(observation, copy=True)
        obs[..., self.mask_indices] = 0.0
        return obs


class PiecewiseDriftWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        seed: int,
        phase_len: int,
        obs_shift_scale: float,
        reward_scale_low: float,
        reward_scale_high: float,
    ):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Box)
        self.rng = np.random.RandomState(seed)
        self.phase_len = int(phase_len)
        self.obs_shift_scale = float(obs_shift_scale)
        self.reward_scale_low = float(reward_scale_low)
        self.reward_scale_high = float(reward_scale_high)

        self.t = 0
        self.phase = 0
        self.shift = np.zeros(env.observation_space.shape, dtype=np.float32)
        self.r_scale = 1.0

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self.t = 0
        self.phase = 0
        self.shift = self.rng.randn(*obs.shape).astype(np.float32) * self.obs_shift_scale
        self.r_scale = self.rng.uniform(self.reward_scale_low, self.reward_scale_high)
        return (obs + self.shift), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.t += 1
        if self.phase_len > 0 and (self.t % self.phase_len) == 0:
            self.phase += 1
            self.shift = self.rng.randn(*obs.shape).astype(np.float32) * self.obs_shift_scale
            self.r_scale = self.rng.uniform(self.reward_scale_low, self.reward_scale_high)
        return (obs + self.shift), (reward * self.r_scale), terminated, truncated, info


class EnvPool:
    def __init__(self, env_fns: list):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space

        self._ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_returns = []

    def reset(self, seed: int):
        obs = []
        infos = []
        for i, env in enumerate(self.envs):
            o, info = env.reset(seed=seed + i)
            obs.append(o)
            infos.append(info)
            self._ep_returns[i] = 0.0
            self._ep_lengths[i] = 0
        return np.stack(obs, axis=0), infos

    def step(self, actions: np.ndarray):
        obs, rew, term, trunc, infos = [], [], [], [], []
        for i, env in enumerate(self.envs):
            o, r, t, tr, info = env.step(int(actions[i]))
            self._ep_returns[i] += float(r)
            self._ep_lengths[i] += 1
            if t or tr:
                self.episode_returns.append((float(self._ep_returns[i]), int(self._ep_lengths[i])))
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0
                o, info_reset = env.reset()
                info["reset_info"] = info_reset
            obs.append(o)
            rew.append(r)
            term.append(t)
            trunc.append(tr)
            infos.append(info)
        return (
            np.stack(obs, axis=0),
            np.asarray(rew, dtype=np.float32),
            np.asarray(term, dtype=np.bool_),
            np.asarray(trunc, dtype=np.bool_),
            infos,
        )
