from concurrent.futures import ThreadPoolExecutor

import numpy as np
import gymnasium as gym

try:
    import albumentations as A
except ImportError:  # pragma: no cover - optional dependency for CarRacing preprocessing
    A = None


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


class DiscreteCarRacingWrapper(gym.ActionWrapper):
    """
    Convert CarRacing's continuous action space [steer, gas, brake] into a
    small discrete action set so discrete-policy algorithms can train.
    """

    DEFAULT_ACTIONS = np.asarray(
        [
            [0.0, 0.0, 0.0],   # coast
            [0.0, 1.0, 0.0],   # gas
            [0.0, 0.0, 0.8],   # brake
            [-0.6, 0.0, 0.0],  # left (no gas)
            [0.6, 0.0, 0.0],   # right (no gas)
            [-0.6, 0.4, 0.0],  # left + soft gas
            [0.6, 0.4, 0.0],   # right + soft gas
        ],
        dtype=np.float32,
    )

    def __init__(self, env: gym.Env, actions: np.ndarray | None = None, *, smooth_beta: float = 0.0):
        super().__init__(env)
        if not isinstance(env.action_space, gym.spaces.Box):
            raise TypeError("DiscreteCarRacingWrapper expects a Box action space.")
        if tuple(env.action_space.shape) != (3,):
            raise ValueError(f"Expected action shape (3,), got {env.action_space.shape}")

        table = self.DEFAULT_ACTIONS if actions is None else np.asarray(actions, dtype=np.float32)
        if table.ndim != 2 or table.shape[1] != 3:
            raise ValueError(f"`actions` must have shape [K, 3], got {table.shape}")

        low = env.action_space.low.astype(np.float32)
        high = env.action_space.high.astype(np.float32)
        self._actions = np.clip(table, low, high).astype(np.float32)
        self.action_space = gym.spaces.Discrete(int(self._actions.shape[0]))
        self.smooth_beta = float(smooth_beta)
        if not (0.0 <= self.smooth_beta <= 1.0):
            raise ValueError(f"`smooth_beta` must be in [0, 1], got {self.smooth_beta}")
        self._prev = np.zeros(3, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._prev[:] = 0.0
        return obs, info

    @property
    def action_table(self) -> np.ndarray:
        return self._actions.copy()

    def action(self, act: int) -> np.ndarray:
        idx = int(act)
        if idx < 0 or idx >= self.action_space.n:
            raise ValueError(f"Discrete action index out of range: {idx}")
        a = self._actions[idx].copy()
        beta = self.smooth_beta
        if beta > 0.0:
            a = ((1.0 - beta) * self._prev + beta * a).astype(np.float32, copy=False)
        self._prev[:] = a
        return a


class CarRacingPreprocessWrapper(gym.ObservationWrapper):
    """Optional preprocessing to speed up CarRacing training.

    - `downsample`: keep every K-th pixel along H/W.
    - `grayscale`: convert RGB -> single channel (uint8).
    """

    def __init__(self, env: gym.Env, *, downsample: int = 1, grayscale: bool = False):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("CarRacingPreprocessWrapper expects a Box observation space.")
        if A is None:
            raise ImportError(
                "CarRacingPreprocessWrapper requires `albumentations`. "
                "Install dependencies from requirements.txt."
            )
        downsample = int(downsample)
        if downsample < 1:
            raise ValueError("--carracing-downsample must be >= 1.")
        self.downsample = downsample
        self.grayscale = bool(grayscale)

        space = env.observation_space
        shape = tuple(space.shape)
        if len(shape) != 3:
            raise ValueError(f"Expected image observation shape (H, W, C), got {shape}")
        h, w, c = shape
        if self.grayscale and c != 3:
            raise ValueError(f"Grayscale conversion expects 3 channels, got C={c}")

        out_h = h // self.downsample
        out_w = w // self.downsample
        if out_h < 1 or out_w < 1:
            raise ValueError(
                f"--carracing-downsample={self.downsample} is too large for input shape {shape}; "
                "resulting image size must be at least 1x1."
            )
        out_c = 1 if self.grayscale else c
        out_shape = (out_h, out_w, out_c)

        low = np.full(out_shape, float(np.min(space.low)), dtype=space.dtype)
        high = np.full(out_shape, float(np.max(space.high)), dtype=space.dtype)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=space.dtype)
        transforms: list = []
        if self.downsample > 1:
            transforms.append(A.Resize(height=out_h, width=out_w, p=1.0))
        if self.grayscale:
            transforms.append(A.ToGray(num_output_channels=1, p=1.0))
        self._transform = A.Compose(transforms)

    def observation(self, observation):
        obs = np.asarray(observation)
        out = self._transform(image=obs)["image"]
        if out.ndim == 2:
            out = out[..., None]
        if out.dtype != self.observation_space.dtype:
            out = out.astype(self.observation_space.dtype, copy=False)
        return out


class FrameStackLastAxisWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise TypeError("FrameStackLastAxisWrapper expects a Box observation space.")
        num_stack = int(num_stack)
        if num_stack < 1:
            raise ValueError("--frame-stack must be >= 1.")
        self.num_stack = num_stack

        space = env.observation_space
        low = np.asarray(space.low)
        high = np.asarray(space.high)
        if low.ndim < 1:
            raise ValueError(f"Frame stack expects at least 1D observations, got shape={space.shape}")
        self.observation_space = gym.spaces.Box(
            low=np.concatenate([low] * self.num_stack, axis=-1),
            high=np.concatenate([high] * self.num_stack, axis=-1),
            dtype=space.dtype,
        )
        self._frames: list[np.ndarray] = []

    def _stack_obs(self) -> np.ndarray:
        return np.concatenate(self._frames, axis=-1)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        first = np.asarray(obs).copy()
        self._frames = [first.copy() for _ in range(self.num_stack)]
        return self._stack_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not self._frames:
            first = np.asarray(obs).copy()
            self._frames = [first.copy() for _ in range(self.num_stack)]
        else:
            self._frames.pop(0)
            self._frames.append(np.asarray(obs).copy())
        return self._stack_obs(), reward, terminated, truncated, info


class EnvPool:
    def __init__(self, env_fns: list, *, workers: int = 0):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.single_observation_space = self.envs[0].observation_space
        self.single_action_space = self.envs[0].action_space
        self.workers = int(workers)
        self._executor = ThreadPoolExecutor(max_workers=self.workers) if self.workers > 1 else None

        self._ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.episode_returns = []

    def reset(self, seed: int):
        obs, infos = [], []
        if self._executor is None:
            results = [env.reset(seed=seed + i) for i, env in enumerate(self.envs)]
        else:
            futures = [self._executor.submit(env.reset, seed=seed + i) for i, env in enumerate(self.envs)]
            results = [f.result() for f in futures]
        for i, (o, info) in enumerate(results):
            obs.append(o)
            infos.append(info)
            self._ep_returns[i] = 0.0
            self._ep_lengths[i] = 0
        return np.stack(obs, axis=0), infos

    def step(self, actions: np.ndarray):
        obs, rew, term, trunc, infos = [], [], [], [], []
        final_obs: dict[int, np.ndarray] = {}

        actions_i = [int(actions[i]) for i in range(self.num_envs)]
        if self._executor is None:
            results = [env.step(actions_i[i]) for i, env in enumerate(self.envs)]
        else:
            futures = [self._executor.submit(env.step, actions_i[i]) for i, env in enumerate(self.envs)]
            results = [f.result() for f in futures]

        done_indices: list[int] = []
        for i, (o, r, t, tr, info) in enumerate(results):
            self._ep_returns[i] += float(r)
            self._ep_lengths[i] += 1
            if t or tr:
                self.episode_returns.append((float(self._ep_returns[i]), int(self._ep_lengths[i])))
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0
                done_indices.append(i)
                final_obs[i] = np.asarray(o).copy()
            obs.append(o)
            rew.append(r)
            term.append(t)
            trunc.append(tr)
            if not isinstance(info, dict):
                info = {"env_info": info}
            infos.append(info)

        if done_indices:
            if self._executor is None:
                for i in done_indices:
                    o_reset, info_reset = self.envs[i].reset()
                    obs[i] = o_reset
                    infos[i]["final_obs"] = final_obs[i]
                    infos[i]["reset_info"] = info_reset
            else:
                reset_futures = {i: self._executor.submit(self.envs[i].reset) for i in done_indices}
                for i, fut in reset_futures.items():
                    o_reset, info_reset = fut.result()
                    obs[i] = o_reset
                    infos[i]["final_obs"] = final_obs[i]
                    infos[i]["reset_info"] = info_reset
        return (
            np.stack(obs, axis=0),
            np.asarray(rew, dtype=np.float32),
            np.asarray(term, dtype=np.bool_),
            np.asarray(trunc, dtype=np.bool_),
            infos,
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass
