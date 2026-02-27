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
    ):
        super().__init__()
        self.encoder_type = str(encoder_type)
        self.obs_shape = tuple(obs_shape) if obs_shape is not None else None
        self.act_emb = nn.Embedding(act_dim, act_embed_dim)
        if self.encoder_type == "mlp":
            self.obs_encoder = MLP(obs_dim, hidden_dim, feat_dim, activation=nn.Tanh)
        elif self.encoder_type == "cnn":
            if self.obs_shape is None:
                raise ValueError("obs_shape is required when encoder_type='cnn'.")
            self.obs_encoder = ConvEncoder(self.obs_shape, hidden_dim=hidden_dim, out_dim=feat_dim)
        else:
            raise ValueError(f"Unsupported encoder_type: {self.encoder_type}")

        self.fuse = MLP(feat_dim + act_embed_dim, hidden_dim, feat_dim, activation=nn.Tanh)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        if self.encoder_type == "mlp" and obs.ndim > 2:
            obs = obs.reshape(obs.shape[0], -1)
        x_obs = self.obs_encoder(obs)
        a = self.act_emb(prev_action.long())
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
    ):
        super().__init__()
        self.f_pol = FeatureEncoder(
            obs_dim,
            act_dim,
            act_embed_dim,
            hidden_dim,
            feat_dim,
            encoder_type=encoder_type,
            obs_shape=obs_shape,
        )
        self.core = MLP(feat_dim + mem_dim, hidden_dim, hidden_dim, activation=nn.Tanh)
        self.pi = nn.Linear(hidden_dim, act_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def forward(
        self, obs: torch.Tensor, prev_action: torch.Tensor, traces_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_pol = self.f_pol(obs, prev_action)
        h = self.core(torch.cat([x_pol, traces_flat], dim=-1))
        logits = self.pi(h)
        value = self.v(h).squeeze(-1)
        return logits, value


class Predictor(nn.Module):
    def __init__(self, feat_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.act_emb = nn.Embedding(act_dim, hidden_dim)
        self.net = MLP(feat_dim + hidden_dim, hidden_dim, feat_dim, activation=nn.Tanh)

    def forward(self, x_mem: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a = self.act_emb(action.long())
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
    ):
        super().__init__()
        self.f_pol = FeatureEncoder(
            obs_dim,
            act_dim,
            act_embed_dim,
            hidden_dim,
            feat_dim,
            encoder_type=encoder_type,
            obs_shape=obs_shape,
        )
        self.lstm = nn.LSTM(input_size=feat_dim, hidden_size=hidden_dim, num_layers=1)
        self.pi = nn.Linear(hidden_dim, act_dim)
        self.v = nn.Linear(hidden_dim, 1)

    def init_hidden(self, batch_size: int, device: str):
        h = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size, device=device)
        return h, c

    def forward(
        self, obs: torch.Tensor, prev_action: torch.Tensor, hidden: tuple[torch.Tensor, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.f_pol(obs, prev_action)  # [B, feat]
        x = x.unsqueeze(0)  # [1, B, feat]
        out, (h, c) = self.lstm(x, hidden)
        out = out.squeeze(0)
        logits = self.pi(out)
        value = self.v(out).squeeze(-1)
        return logits, value, (h, c)
