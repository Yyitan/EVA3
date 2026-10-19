import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAExpert(nn.Module):
    """A LoRA-based expert: E_i(x) = B_i * act(A_i * x).

    Matches the paper: each expert has two trainable matrices A in R^{r x d}
    and B in R^{d x r} (r << d), implemented as a two-layer MLP with a hidden
    dim r. Standard LoRA init: A is kaiming-uniform, B is zero so that the
    adapter outputs ~0 at the beginning (training starts from frozen CLIP).
    """

    def __init__(self, d_model, r=64, scale=1.0, dropout=0.0, act_layer=nn.GELU):
        super().__init__()
        self.down = nn.Linear(d_model, r, bias=False)   # A: d -> r
        self.act = act_layer()
        self.drop = nn.Dropout(dropout)                 # regularization
        self.up = nn.Linear(r, d_model, bias=False)     # B: r -> d
        self.scale = scale                              # LoRA-style output scaling
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return self.scale * self.up(self.drop(self.act(self.down(x))))


class Router(nn.Module):
    """Single linear layer producing token-to-expert affinity logits."""

    def __init__(self, d_model, n_routed):
        super().__init__()
        self.gate = nn.Linear(d_model, n_routed, bias=False)
        nn.init.kaiming_uniform_(self.gate.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.gate(x)


class MoEAdapter(nn.Module):
    """Token-wise MoE adapter, parallel to the FFN of a transformer block.

    Composition (paper Eq.1-3):
        G = Softmax(TopK(R(h)))            # gate over the K selected experts
        h_out = sum_i G_i * E_i(h) + E_0(h)

    * E_0 is the shared expert (always active).
    * E_1..E_Ne are routed experts; only the top-K are active per token.
    """

    def __init__(self, d_model, n_routed=8, k=2, r=64, scale=1.0, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.n_routed = n_routed
        self.k = k
        self.shared = LoRAExpert(d_model, r, scale, dropout)
        self.experts = nn.ModuleList([LoRAExpert(d_model, r, scale, dropout) for _ in range(n_routed)])
        self.router = Router(d_model, n_routed)

    def _gates(self, x):
        """Return dense gate weights [..., n_routed] with non-topk set to 0."""
        logits = self.router(x)                                   # [..., n_routed]
        topk_val, topk_idx = logits.topk(self.k, dim=-1)          # [..., k]
        topk_w = F.softmax(topk_val, dim=-1)                      # softmax over the K selected
        gates = torch.zeros_like(logits)
        gates.scatter_(-1, topk_idx, topk_w)
        return gates

    def forward(self, x):
        """x: [..., d] -> [..., d].

        Sparse dispatch: each routed expert only processes the tokens routed to
        it (top-K), which keeps both compute and activation memory ~ K/Ne of a
        dense implementation. The shared expert always processes all tokens.
        """
        shape = x.shape
        flat = x.reshape(-1, self.d_model)                       # [T, d]
        gates = self._gates(flat)                                # [T, n_routed], 0 except topk
        out = self.shared(flat)                                  # [T, d]
        for i, expert in enumerate(self.experts):
            mask = gates[:, i] > 0
            if mask.any():
                idx = mask.nonzero(as_tuple=True)[0]             # [m]
                sel = flat.index_select(0, idx)                  # [m, d]
                eo = expert(sel) * gates.index_select(0, idx)[:, i:i + 1]
                out = out.index_add(0, idx, eo)
        return out.reshape(shape)

    def expert_outputs(self, x):
        """Per-expert outputs (no routing/gating), used to build image variants.

        Returns a tensor of shape [..., n_routed + 1, d] where index 0 is the
        shared expert and indices 1..Ne are the routed experts.
        """
        outs = [self.shared(x)]                                   # [..., d]
        for expert in self.experts:
            outs.append(expert(x))
        return torch.stack(outs, dim=-2)                          # [..., Ne+1, d]
