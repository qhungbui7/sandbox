import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, activation=nn.Tanh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvEncoder(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], hidden_dim: int, out_dim: int):
        super().__init__()
        if len(obs_shape) != 3:
            raise ValueError(f"CNN encoder expects image obs shape (H, W, C), got {obs_shape}")
        h, w, c = (int(obs_shape[0]), int(obs_shape[1]), int(obs_shape[2]))
        if min(h, w, c) <= 0:
            raise ValueError(f"Invalid image obs shape: {obs_shape}")

        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            conv_out = self.conv(dummy)
            flat_dim = int(conv_out.reshape(1, -1).shape[-1])
        self.head = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"CNN encoder expects obs [B,H,W,C], got shape={tuple(obs.shape)}")
        x = obs.permute(0, 3, 1, 2).contiguous().float()
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
        return self.head(x)


class FeatureEncoder(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_embed_dim: int,
        hidden_dim: int,
        feat_dim: int,
        *,
        encoder_type: str = "mlp",
        obs_shape: tuple[int, ...] | None = None,
        action_type: str = "discrete",
        use_prev_action: bool = True,
    ):
        super().__init__()
        self.encoder_type = str(encoder_type)
        self.obs_shape = tuple(obs_shape) if obs_shape is not None else None
        self.action_type = str(action_type)
        self.act_dim = int(act_dim)
        self.use_prev_action = bool(use_prev_action)
        if self.use_prev_action:
            if self.action_type == "discrete":
                self.act_emb = nn.Embedding(act_dim, act_embed_dim)
                self.act_proj = None
            elif self.action_type == "continuous":
                self.act_emb = None
                self.act_proj = MLP(self.act_dim, hidden_dim, act_embed_dim, activation=nn.Tanh)
            else:
                raise ValueError(f"Unsupported action_type: {self.action_type}")
        else:
            self.act_emb = None
            self.act_proj = None
        if self.encoder_type == "mlp":
            self.obs_encoder = MLP(obs_dim, hidden_dim, feat_dim, activation=nn.Tanh)
        elif self.encoder_type == "cnn":
            if self.obs_shape is None:
                raise ValueError("obs_shape is required when encoder_type='cnn'.")
            self.obs_encoder = ConvEncoder(self.obs_shape, hidden_dim=hidden_dim, out_dim=feat_dim)
        else:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")

        if self.use_prev_action:
            self.fuse = MLP(feat_dim + act_embed_dim, hidden_dim, feat_dim, activation=nn.Tanh)
        else:
            self.fuse = None

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor | None = None) -> torch.Tensor:
        if self.encoder_type == "mlp" and obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        x_obs = self.obs_encoder(obs)
        if not self.use_prev_action:
            return x_obs
        if prev_action is None:
            raise ValueError("prev_action is required when use_prev_action=True.")
        if self.action_type == "discrete":
            a = self.act_emb(prev_action.long())
        else:
            a_in = prev_action.float()
            if a_in.ndim > 2:
                a_in = a_in.reshape(a_in.shape[0], -1)
            elif a_in.ndim == 1:
                a_in = a_in.reshape(a_in.shape[0], 1)
            a = self.act_proj(a_in)
        x = torch.cat([x_obs, a], dim=-1)
        return self.fuse(x)


class ActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_embed_dim: int,
        hidden_dim: int,
        feat_dim: int,
        mem_dim: int,
        *,
        encoder_type: str = "mlp",
        obs_shape: tuple[int, ...] | None = None,
        action_type: str = "discrete",
        use_prev_action: bool = True,
        use_traces: bool = True,
    ):
        super().__init__()
        self.action_type = str(action_type)
        self.act_dim = int(act_dim)
        self.use_traces = bool(use_traces)
        self.f_pol = FeatureEncoder(
            obs_dim,
            act_dim,
            act_embed_dim,
            hidden_dim,
            feat_dim,
            encoder_type=encoder_type,
            obs_shape=obs_shape,
            action_type=self.action_type,
            use_prev_action=use_prev_action,
        )
        core_in_dim = int(feat_dim + (mem_dim if self.use_traces else 0))
        self.core = MLP(core_in_dim, hidden_dim, hidden_dim, activation=nn.Tanh)
        if self.action_type == "discrete":
            self.pi = nn.Linear(hidden_dim, act_dim)
        elif self.action_type == "continuous":
            self.pi_mean = nn.Linear(hidden_dim, act_dim)
            self.pi_log_std = nn.Parameter(torch.zeros(int(act_dim), dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")
        self.v = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor | None = None,
        traces_flat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_pol = self.f_pol(obs, prev_action)
        if self.use_traces:
            if traces_flat is None:
                raise ValueError("traces_flat is required when use_traces=True.")
            core_in = torch.cat([x_pol, traces_flat], dim=-1)
        else:
            core_in = x_pol
        h = self.core(core_in)
        if self.action_type == "discrete":
            logits = self.pi(h)
        else:
            mean = self.pi_mean(h)
            log_std = self.pi_log_std.expand_as(mean)
            logits = torch.cat([mean, log_std], dim=-1)
        value = self.v(h).squeeze(-1)
        return logits, value


class Predictor(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, hidden_dim: int, *, action_type: str = "discrete"):
        super().__init__()
        self.action_type = str(action_type)
        self.act_dim = int(act_dim)
        if self.action_type == "discrete":
            self.act_emb = nn.Embedding(act_dim, hidden_dim)
            self.act_proj = None
        elif self.action_type == "continuous":
            self.act_emb = None
            self.act_proj = MLP(self.act_dim, hidden_dim, hidden_dim, activation=nn.Tanh)
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")
        self.net = MLP(feat_dim + hidden_dim, hidden_dim, feat_dim, activation=nn.Tanh)

    def forward(self, x_mem: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if self.action_type == "discrete":
            a = self.act_emb(action.long())
        else:
            a_in = action.float()
            if a_in.ndim > 2:
                a_in = a_in.reshape(a_in.shape[0], -1)
            elif a_in.ndim == 1:
                a_in = a_in.reshape(a_in.shape[0], 1)
            a = self.act_proj(a_in)
        return self.net(torch.cat([x_mem, a], dim=-1))


class RecurrentActorCritic(nn.Module):
    """PPO policy with an LSTM core (no external trace memory)."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_embed_dim: int,
        hidden_dim: int,
        feat_dim: int,
        *,
        encoder_type: str = "mlp",
        obs_shape: tuple[int, ...] | None = None,
        action_type: str = "discrete",
        use_prev_action: bool = True,
    ):
        super().__init__()
        self.action_type = str(action_type)
        self.act_dim = int(act_dim)
        self.f_pol = FeatureEncoder(
            obs_dim,
            act_dim,
            act_embed_dim,
            hidden_dim,
            feat_dim,
            encoder_type=encoder_type,
            obs_shape=obs_shape,
            action_type=self.action_type,
            use_prev_action=use_prev_action,
        )
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden_dim, num_layers=1)
        if self.action_type == "discrete":
            self.pi = nn.Linear(hidden_dim, act_dim)
        elif self.action_type == "continuous":
            self.pi_mean = nn.Linear(hidden_dim, act_dim)
            self.pi_log_std = nn.Parameter(torch.zeros(int(act_dim), dtype=torch.float32))
        else:
            raise ValueError(f"Unsupported action_type: {self.action_type}")
        self.v = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: str):
        h = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return h, c

    def forward(
        self,
        obs: torch.Tensor,
        prev_action: torch.Tensor | None,
        hidden: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.f_pol(obs, prev_action)  # [B, feat]
        x = x.unsqueeze(0)  # [1, B, feat]
        out, (h, c) = self.lstm(x, hidden)
        out = out.squeeze(0)
        if self.action_type == "discrete":
            logits = self.pi(out)
        else:
            mean = self.pi_mean(out)
            log_std = self.pi_log_std.expand_as(mean)
            logits = torch.cat([mean, log_std], dim=-1)
        value = self.v(out).squeeze(-1)
        return logits, value, (h, c)
