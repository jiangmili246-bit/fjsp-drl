import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class TernaryHGATEncoder(nn.Module):
    """Ternary heterogeneous encoder for job-operation-machine graph."""
    def __init__(self, job_in, ope_in, ma_in, hidden_dim=64, num_heads=4):
        super().__init__()
        self.job_proj = nn.Linear(job_in, hidden_dim)
        self.ope_proj = nn.Linear(ope_in, hidden_dim)
        self.ma_proj = nn.Linear(ma_in, hidden_dim)
        self.jo_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.om_attn = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)

    def forward(self, job_x, ope_x, ma_x, ready_opes_mask, legal_om_mask):
        h_j = self.job_proj(job_x)
        h_o = self.ope_proj(ope_x)
        h_m = self.ma_proj(ma_x)

        # J-O relation attention: active job info flows to active/ready operations
        jo_key_padding = ~ready_opes_mask
        h_o2, _ = self.jo_attn(h_o, h_j, h_j, key_padding_mask=None)
        h_o = h_o + h_o2

        # O-M relation attention: candidate O-M relation
        h_m2, _ = self.om_attn(h_m, h_o, h_o)
        h_m = h_m + h_m2

        g = torch.cat([h_j.mean(dim=1), h_o.mean(dim=1), h_m.mean(dim=1)], dim=-1)
        return h_j, h_o, h_m, g


class HGATActor(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h_o, h_m, g, legal_mask):
        b, o, d = h_o.shape
        m = h_m.size(1)
        o_pad = h_o.unsqueeze(2).expand(-1, -1, m, -1)
        m_pad = h_m.unsqueeze(1).expand(-1, o, -1, -1)
        g_pad = g[:, None, None, :].expand(-1, o, m, -1)
        x = torch.cat([o_pad, m_pad, g_pad], dim=-1)
        logits = self.scorer(x).squeeze(-1)
        logits[~legal_mask] = float('-inf')
        probs = F.softmax(logits.flatten(1), dim=1)
        return probs


class HGATCritic(nn.Module):
    def __init__(self, graph_dim, hidden_dim=64):
        super().__init__()
        self.value = nn.Sequential(nn.Linear(graph_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, g):
        return self.value(g).squeeze(-1)


class HGATPPO(nn.Module):
    def __init__(self, model_paras, train_paras):
        super().__init__()
        self.device = model_paras['device']
        self.gamma = train_paras['gamma']
        self.eps_clip = train_paras['eps_clip']
        self.K_epochs = train_paras['K_epochs']
        self.entropy_coeff = train_paras['entropy_coeff']
        self.vf_coeff = train_paras['vf_coeff']
        self.encoder = TernaryHGATEncoder(8, 3, 4, hidden_dim=model_paras.get('hgat_hidden_dim', 64),
                                          num_heads=model_paras.get('hgat_heads', 4)).to(self.device)
        h = model_paras.get('hgat_hidden_dim', 64)
        self.actor = HGATActor(h).to(self.device)
        self.critic = HGATCritic(h * 3, h).to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=train_paras['lr'])
        self.policy_old = copy.deepcopy(self)

    def act(self, state):
        job_x = state.feat_job_norm_batch.transpose(1, 2)
        ope_x = state.feat_opes_batch.transpose(1, 2)
        ma_x = state.feat_mas_batch.transpose(1, 2)
        ready = state.ready_opes_batch
        legal = state.legal_action_mask_batch
        _, h_o, h_m, g = self.encoder(job_x, ope_x, ma_x, ready, legal)
        probs = self.actor(h_o, h_m, g, legal)
        dist = Categorical(probs)
        a_idx = dist.sample()
        num_opes = ope_x.size(1)
        mas = (a_idx / num_opes).long()
        opes = (a_idx % num_opes).long()
        jobs = state.opes_appertain_batch[state.batch_idxes, opes]
        return torch.stack((opes, mas, jobs), dim=1).t(), dist.log_prob(a_idx), dist.entropy()
