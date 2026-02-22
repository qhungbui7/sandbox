import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FeatureEncoder(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_embed_dim: int, hidden_dim: int, feat_dim: int):
        super().__init__()
        self.act_emb = nn.Embedding(act_dim, act_embed_dim)
        self.mlp = MLP(obs_dim + act_embed_dim, hidden_dim, feat_dim, activation=nn.Tanh)

    def forward(self, obs: torch.Tensor, prev_action: torch.Tensor) -> torch.Tensor:
        a = self.act_emb(prev_action.long())
        x = torch.cat([obs, a], dim=-1)
        return self.mlp(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, act_embed_dim: int, hidden_dim: int, feat_dim: int, mem_dim: int):
        super().__init__()
        self.f_pol = FeatureEncoder(obs_dim, act_dim, act_embed_dim, hidden_dim, feat_dim)
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
    def __init__(self, feat_dim: int, act_dim: int):
        super().__init__()
        self.act_emb = nn.Embedding(act_dim, feat_dim)
        self.mlp = MLP(feat_dim + feat_dim, feat_dim, feat_dim, activation=nn.Tanh)

    def forward(self, x_mem: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        a = self.act_emb(action.long())
        x = torch.cat([x_mem, a], dim=-1)
        return self.mlp(x)


class RecurrentActorCritic(nn.Module):
    """PPO policy with an LSTM core (no external trace memory)."""

    def __init__(self, obs_dim: int, act_dim: int, act_embed_dim: int, hidden_dim: int, feat_dim: int):
        super().__init__()
        self.f_pol = FeatureEncoder(obs_dim, act_dim, act_embed_dim, hidden_dim, feat_dim)
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
